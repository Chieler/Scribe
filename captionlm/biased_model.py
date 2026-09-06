"""BiasedParakeetTDTCTC: CTC-WS context-biasing on top of parakeet-mlx's
hybrid TDT-CTC model. Runs the CTC head's log-probs through the vendored
word spotter, then merges spotted terms into the TDT hypothesis by
frame-overlap. When .context_graph is None, behaves exactly like the
stock ParakeetTDTCTC.generate()."""
import json

import mlx.core as mx
import numpy as np
from dacite import from_dict
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten
from parakeet_mlx import tokenizer
from parakeet_mlx.alignment import (
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.parakeet import DecodingConfig, ParakeetTDTCTC, ParakeetTDTCTCArgs

from captionlm.config import SpotterConfig
from captionlm.merge import merge_spans
from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import run_word_spotter


class BiasedParakeetTDTCTC(ParakeetTDTCTC):
    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)
        self.context_graph: ContextGraphCTC | None = None
        self.spotter_config = SpotterConfig()
        # Only populated when self.context_graph is set -- capture_logits
        # is appended to inside generate()'s context-graph branch, so an
        # unbiased call (context_graph is None) captures nothing.
        self.capture_logits: list[np.ndarray] | None = None

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ):
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)
        return self._hypotheses(features, lengths, decoding_config)

    def generate_batch(
        self,
        mels: list[mx.array],
        *,
        decoding_config: DecodingConfig = DecodingConfig(),
    ):
        """Transcribe several independent chunks under one decode loop.

        The encoder still runs per chunk: it is compute-bound, and batching
        it buys nothing (measured 0.41 s per 120 s chunk at any batch size).
        The TDT decode loop is the opposite -- one step costs 0.49 ms at
        batch 1 and 0.71 ms at batch 8, because it is dispatch-bound -- so
        chunks decoded side by side are close to free. Chunks were always
        independent (`transcribe` merges them only afterwards), so this
        returns exactly what generate() per chunk returns; the batched
        decode is checked against the sequential one in the tests.
        """
        features, lengths = [], []
        for mel in mels:
            if len(mel.shape) == 2:
                mel = mx.expand_dims(mel, 0)
            chunk_features, chunk_lengths = self.encoder(mel)
            mx.eval(chunk_features, chunk_lengths)
            features.append(chunk_features)
            lengths.append(int(chunk_lengths[0]))

        width = max(f.shape[1] for f in features)
        padded = mx.concatenate(
            [mx.pad(f, [(0, 0), (0, width - f.shape[1]), (0, 0)]) for f in features]
        )
        # The padding is never decoded: every step index is clamped to its own
        # chunk's length below.
        return self._hypotheses(padded, mx.array(lengths), decoding_config)

    def _hypotheses(
        self, features: mx.array, lengths: mx.array, decoding_config: DecodingConfig
    ):
        result, _ = self.decode(features, lengths, config=decoding_config)

        if self.context_graph is None:
            hypotheses = result
        else:
            ctc_logits = self.ctc_decoder(features)
            mx.eval(ctc_logits)
            blank_idx = len(self.vocabulary)

            hypotheses = []
            for batch_idx, tokens in enumerate(result):
                # Trimmed to this chunk's own length: a batch of chunks is
                # padded to the longest of them, and spotting inside another
                # chunk's padding would invent terms out of silence.
                frames = int(lengths[batch_idx])
                logprobs = np.array(ctc_logits[batch_idx, :frames].astype(mx.float32))
                if self.capture_logits is not None:
                    self.capture_logits.append(logprobs.copy())
                ws_hyps = run_word_spotter(
                    logprobs,
                    self.context_graph,
                    self,
                    blank_idx=blank_idx,
                    beam_threshold=self.spotter_config.beam_threshold,
                    cb_weight=self.spotter_config.cb_weight,
                    ctc_ali_token_weight=self.spotter_config.ctc_ali_token_weight,
                    keyword_threshold=self.spotter_config.keyword_threshold,
                    blank_threshold=self.spotter_config.blank_threshold,
                    non_blank_threshold=self.spotter_config.non_blank_threshold,
                )
                hypotheses.append(
                    merge_spans(
                        tokens,
                        ws_hyps,
                        self.time_ratio,
                        intersection_threshold=self.spotter_config.intersection_threshold,
                    )
                )

        return [
            sentences_to_result(tokens_to_sentences(h, decoding_config.sentence))
            for h in hypotheses
        ]

    def decode_greedy(
        self,
        features: mx.array,
        lengths: mx.array | None = None,
        last_token: list[int | None] | None = None,
        hidden_state: list[tuple[mx.array, mx.array] | None] | None = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ):
        """The base class's TDT greedy loop, decoding the whole batch at once.

        The base class walks the batch one item at a time and synchronises the
        GPU three times per step (token argmax, duration argmax, confidence).
        The step is dispatch-bound rather than compute-bound, so both cost
        real time: folding the syncs into one is worth ~30% at batch 1, and
        decoding the batch together turns N sequential loops into one.

        Emitted tokens are identical to the base class's, which is what the
        tests assert. Streaming state (`last_token` / `hidden_state`) is left
        to the base class -- `StreamingParakeet` passes it, this project does
        not, and it is per-item state that does not survive being batched.
        """
        if last_token is not None or hidden_state is not None:
            return super().decode_greedy(
                features, lengths, last_token, hidden_state, config=config
            )

        batch, width, *_ = features.shape
        if lengths is None:
            lengths = mx.array([width] * batch)
        limits = [int(lengths[b]) for b in range(batch)]
        blank = len(self.vocabulary)
        embed = self.decoder.prediction["embed"]
        # Driven layer by layer rather than through the LSTM wrapper: the
        # wrapper transposes to (1, B, P), which mlx's nn.LSTM reads as one
        # sequence of B timesteps -- correct only at batch 1.
        layers = self.decoder.prediction["dec_rnn"].lstm

        hypotheses: list[list[AlignedToken]] = [[] for _ in range(batch)]
        steps = [0] * batch
        last = [None] * batch
        new_symbols = [0] * batch
        hidden = cell = None

        while any(steps[b] < limits[b] for b in range(batch)):
            embedded = embed(mx.array([[0 if t is None else t] for t in last]))
            if any(t is None for t in last):
                # An item that has emitted nothing yet gets the zero embedding
                # the base class passes as `y=None`.
                started = mx.array(
                    [[[0.0]] if t is None else [[1.0]] for t in last]
                )
                embedded = embedded * started

            out = embedded
            next_hidden, next_cell = [], []
            for i, layer in enumerate(layers):
                out, cells = layer(
                    out,
                    hidden=None if hidden is None else hidden[i],
                    cell=None if cell is None else cell[i],
                )
                next_hidden.append(out[:, -1])
                next_cell.append(cells[:, -1])
            new_hidden = mx.stack(next_hidden)
            new_cell = mx.stack(next_cell)
            decoder_out = out.astype(features.dtype)

            frame = mx.array([min(steps[b], limits[b] - 1) for b in range(batch)])
            joint_out = self.joint(
                mx.take_along_axis(features, frame.reshape(batch, 1, 1), axis=1),
                decoder_out,
            )
            token_logits = joint_out[:, 0, 0, : blank + 1]
            predicted = mx.argmax(token_logits, axis=-1)
            decisions = mx.argmax(joint_out[:, 0, 0, blank + 1 :], axis=-1)
            probs = mx.softmax(token_logits, axis=-1)
            entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=-1)
            confidences = 1.0 - entropy / mx.log(mx.array(blank + 1, dtype=probs.dtype))
            mx.eval(predicted, decisions, confidences, new_hidden, new_cell)

            predicted = predicted.tolist()
            decisions = decisions.tolist()
            confidences = confidences.tolist()
            emitted = [
                steps[b] < limits[b] and predicted[b] != blank for b in range(batch)
            ]

            # Only an item that emitted advances its prediction state, exactly
            # as in the base class; the rest carry theirs forward.
            keep = mx.array([1.0 if e else 0.0 for e in emitted]).reshape(1, batch, 1)
            if hidden is None:
                hidden, cell = new_hidden * keep, new_cell * keep
            else:
                hidden = new_hidden * keep + hidden * (1 - keep)
                cell = new_cell * keep + cell * (1 - keep)
            hidden = hidden.astype(features.dtype)
            cell = cell.astype(features.dtype)

            for b in range(batch):
                if steps[b] >= limits[b]:
                    continue
                duration = self.durations[decisions[b]]
                if emitted[b]:
                    hypotheses[b].append(
                        AlignedToken(
                            predicted[b],
                            start=steps[b] * self.time_ratio,
                            duration=duration * self.time_ratio,
                            confidence=confidences[b],
                            text=tokenizer.decode([predicted[b]], self.vocabulary),
                        )
                    )
                    last[b] = predicted[b]

                steps[b] += duration
                new_symbols[b] += 1
                if duration != 0:
                    new_symbols[b] = 0
                elif self.max_symbols is not None and self.max_symbols <= new_symbols[b]:
                    steps[b] += 1
                    new_symbols[b] = 0

        return hypotheses, [
            (hidden[:, b : b + 1], cell[:, b : b + 1]) for b in range(batch)
        ]


def load_biased_model(
    hf_id: str, *, dtype: mx.Dtype = mx.bfloat16
) -> BiasedParakeetTDTCTC:
    config = json.load(open(hf_hub_download(hf_id, "config.json"), "r"))
    weight = hf_hub_download(hf_id, "model.safetensors")

    cfg = from_dict(ParakeetTDTCTCArgs, config)
    model = BiasedParakeetTDTCTC(cfg)
    model.eval()
    model.load_weights(weight)

    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
