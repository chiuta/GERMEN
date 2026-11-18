//! Text generation: sampling the next token from the model, one token at a time.
//!
//! The model keeps a key/value cache, so every new token costs one token's worth of computation
//! (not a pass over the whole window); when the cache is full, generation goes on from the most
//! recent half of the context. Sampling is plain temperature and top-k over the logits.

use crate::data::Rng;
use crate::model::{Gpt, KvCache};
use crate::tokenizer::{MAX_VOCAB, TokenId, Tokenizer};
use burn::prelude::*;

/// Sampling settings.
#[derive(Clone, Debug)]
pub struct SampleOptions {
    /// Number of new tokens (bytes) to produce.
    pub max_new: usize,
    /// Softmax temperature. `0` (or less) means greedy decoding; higher means more random.
    pub temperature: f32,
    /// Only the `top_k` most likely tokens are considered; `0` disables the filter.
    pub top_k: usize,
    /// Context the model reads at most (clamped to the model's context length). When it is
    /// full, generation goes on from its most recent half.
    pub window: usize,
}

impl Default for SampleOptions {
    fn default() -> Self {
        SampleOptions {
            max_new: 200,
            temperature: 0.8,
            top_k: 40,
            window: 128,
        }
    }
}

/// Chooses a token from raw `logits`.
///
/// Greedy when `temperature <= 0` or `top_k == 1`; otherwise samples from the softmax of
/// `logits / temperature` restricted to the `top_k` highest logits.
pub fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> TokenId {
    assert!(
        !logits.is_empty() && logits.len() <= MAX_VOCAB,
        "expected 1..={MAX_VOCAB} logits"
    );

    let argmax = || {
        let mut best = 0;
        for (i, &l) in logits.iter().enumerate() {
            if l > logits[best] {
                best = i;
            }
        }
        best as TokenId
    };
    if temperature <= 0.0 || top_k == 1 {
        return argmax();
    }

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if top_k > 0 && top_k < order.len() {
        order.truncate(top_k);
    }

    let max = logits[order[0]];
    let weights: Vec<f32> = order
        .iter()
        .map(|&i| ((logits[i] - max) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return argmax();
    }
    let mut draw = rng.next_f32() * total;
    for (k, w) in weights.iter().enumerate() {
        draw -= w;
        if draw <= 0.0 {
            return order[k] as TokenId;
        }
    }
    order[order.len() - 1] as TokenId
}

/// Logits of the token that follows `tokens`, which are read as the continuation of `cache`.
fn next_logits(model: &Gpt, tokens: &[TokenId], cache: &mut KvCache, device: &Device) -> Vec<f32> {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let t = ids.len();
    let x = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, t]), device);
    let logits = model.forward_cached(x, cache); // [1, t, vocab]
    let vocab = logits.dims()[2];
    logits
        .slice([0..1, t - 1..t, 0..vocab])
        .reshape([vocab])
        .into_data()
        .try_to_vec::<f32>()
        .expect("logits are f32")
}

/// Most tokens `generate` reserves room for before producing any.
const MAX_UPFRONT_TOKENS: usize = 1 << 16;

/// Continues `prompt` (text bytes) by `opts.max_new` tokens. `on_token` is called with each new
/// token as soon as it is chosen (use it to stream the text, see
/// [`crate::tokenizer::Utf8Stream::push_token`]). An empty prompt starts from a newline.
///
/// The model keeps a key/value cache, so each new token costs one token's worth of computation.
/// When the cache holds `opts.window` tokens, generation continues from the most recent half of
/// them (one prompt-sized forward pass every `window / 2` tokens).
///
/// Returns only the newly generated tokens; [`Tokenizer::decode_complete`] turns them into text.
pub fn generate(
    model: &Gpt,
    tokenizer: &Tokenizer,
    prompt: &[u8],
    opts: &SampleOptions,
    rng: &mut Rng,
    device: &Device,
    on_token: impl FnMut(TokenId),
) -> Vec<TokenId> {
    generate_until(
        model,
        tokenizer,
        prompt,
        opts,
        rng,
        device,
        || false,
        on_token,
    )
}

/// [`generate`] that asks `stop` before every new token and ends the run, with the tokens made so
/// far, as soon as it answers `true` (a "Stop" button; Ctrl-C).
///
/// The first answer is asked after the prompt has been read, so a very long prompt is not
/// interrupted half-way through; with `max_new == 0` nothing is asked at all.
#[allow(clippy::too_many_arguments)]
pub fn generate_until(
    model: &Gpt,
    tokenizer: &Tokenizer,
    prompt: &[u8],
    opts: &SampleOptions,
    rng: &mut Rng,
    device: &Device,
    mut stop: impl FnMut() -> bool,
    mut on_token: impl FnMut(TokenId),
) -> Vec<TokenId> {
    let mut context: Vec<TokenId> =
        tokenizer.encode(if prompt.is_empty() { b"\n" } else { prompt });
    let window = opts.window.clamp(2, model.block_size());
    // Never reserve more than a modest amount up front: `--max-new` is a number typed by a person
    // (a mistyped one used to abort the process with "capacity overflow" before the first token).
    let mut out = Vec::with_capacity(opts.max_new.min(MAX_UPFRONT_TOKENS));
    if opts.max_new == 0 {
        return out;
    }

    let mut cache = model.new_cache();
    let from = context.len().saturating_sub(window);
    let mut logits = next_logits(model, &context[from..], &mut cache, device);
    for i in 0..opts.max_new {
        if stop() {
            break;
        }
        let next = sample_logits(&logits, opts.temperature, opts.top_k, rng);
        context.push(next);
        out.push(next);
        on_token(next);
        if i + 1 == opts.max_new {
            break;
        }
        logits = if cache.len() >= window {
            cache.clear();
            let keep = (window / 2).max(1);
            next_logits(model, &context[context.len() - keep..], &mut cache, device)
        } else {
            next_logits(model, &[next], &mut cache, device)
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;

    #[test]
    fn greedy_and_top1_pick_the_largest_logit() {
        let mut rng = Rng::new(1);
        let logits = [0.1, 2.5, -1.0, 2.4];
        assert_eq!(sample_logits(&logits, 0.0, 0, &mut rng), 1);
        assert_eq!(sample_logits(&logits, 1.0, 1, &mut rng), 1);
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let mut rng = Rng::new(5);
        // Strongly peaked: index 0 should win almost always.
        let peaked = [10.0, 0.0, 0.0];
        let wins = (0..300)
            .filter(|_| sample_logits(&peaked, 1.0, 0, &mut rng) == 0)
            .count();
        assert!(wins > 290, "wins = {wins}");
        // Uniform: every index should appear.
        let flat = [0.0; 4];
        let mut seen = [false; 4];
        for _ in 0..200 {
            seen[sample_logits(&flat, 1.0, 0, &mut rng) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn top_k_excludes_unlikely_tokens() {
        let mut rng = Rng::new(9);
        let logits = [5.0, 4.9, -3.0, -3.0, -3.0];
        for _ in 0..300 {
            let t = sample_logits(&logits, 2.0, 2, &mut rng);
            assert!(t < 2, "token {t} is outside the top 2");
        }
    }

    #[test]
    fn an_absurd_max_new_does_not_abort_before_the_first_token() {
        // `Vec::with_capacity(usize::MAX)` panics ("capacity overflow"), and anything merely huge
        // aborts the process. The run is cut short from inside the callback, as Ctrl-C would.
        let device = Device::flex();
        let model = Preset::Nano.config(16).init_seeded(&device, 1);
        for max_new in [usize::MAX, usize::MAX / 8, 1usize << 40] {
            let opts = SampleOptions {
                max_new,
                temperature: 1.0,
                top_k: 0,
                window: 8,
            };
            let mut produced = 0;
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate(
                    &model,
                    &Tokenizer::Byte,
                    b"x",
                    &opts,
                    &mut Rng::new(1),
                    &device,
                    |_| {
                        produced += 1;
                        assert!(produced < 5, "enough tokens seen");
                    },
                )
            }));
            assert!(
                outcome.is_err(),
                "the callback must have been reached and stopped the run"
            );
            assert_eq!(produced, 5, "max_new = {max_new}");
        }
    }

    #[test]
    fn generation_ends_as_soon_as_stop_says_so() {
        let device = Device::flex();
        let model = Preset::Nano.config(16).init_seeded(&device, 1);
        let opts = SampleOptions {
            max_new: 50,
            temperature: 1.0,
            top_k: 0,
            window: 8,
        };
        for allowed in [0usize, 1, 7, 49] {
            let mut asked = 0;
            let mut streamed = Vec::new();
            let out = generate_until(
                &model,
                &Tokenizer::Byte,
                b"Ana",
                &opts,
                &mut Rng::new(3),
                &device,
                || {
                    asked += 1;
                    asked > allowed
                },
                |t| streamed.push(t),
            );
            assert_eq!(out.len(), allowed, "allowed = {allowed}");
            assert_eq!(out, streamed);
            assert_eq!(
                asked,
                allowed + 1,
                "asked once before every token, and once more"
            );
        }
    }

    #[test]
    fn a_stop_that_never_fires_changes_nothing() {
        let device = Device::flex();
        let model = Preset::Nano.config(16).init_seeded(&device, 1);
        let opts = SampleOptions {
            max_new: 30,
            temperature: 0.9,
            top_k: 20,
            window: 8,
        };
        let plain = generate(
            &model,
            &Tokenizer::Byte,
            b"Era o dat\xc4\x83",
            &opts,
            &mut Rng::new(11),
            &device,
            |_| {},
        );
        let guarded = generate_until(
            &model,
            &Tokenizer::Byte,
            b"Era o dat\xc4\x83",
            &opts,
            &mut Rng::new(11),
            &device,
            || false,
            |_| {},
        );
        assert_eq!(plain, guarded);
    }

    #[test]
    fn nothing_is_asked_when_nothing_is_to_be_made() {
        let device = Device::flex();
        let model = Preset::Nano.config(16).init_seeded(&device, 1);
        let opts = SampleOptions {
            max_new: 0,
            ..SampleOptions::default()
        };
        let out = generate_until(
            &model,
            &Tokenizer::Byte,
            b"x",
            &opts,
            &mut Rng::new(1),
            &device,
            || panic!("stop must not be asked when max_new is 0"),
            |_| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn generate_returns_requested_length_and_streams_tokens() {
        let device = Device::flex();
        let model = Preset::Nano.config(16).init_seeded(&device, 1);
        let opts = SampleOptions {
            max_new: 12,
            temperature: 1.0,
            top_k: 0,
            window: 8,
        };
        let mut rng = Rng::new(3);
        let mut streamed = Vec::new();
        let out = generate(
            &model,
            &Tokenizer::Byte,
            b"Ana citeste o carte veche",
            &opts,
            &mut rng,
            &device,
            |b| streamed.push(b),
        );
        assert_eq!(out.len(), 12);
        assert_eq!(out, streamed);
    }

    #[test]
    fn generation_far_beyond_the_window_slides_without_trouble() {
        let device = Device::flex();
        let model = Preset::Nano.config(32).init_seeded(&device, 1);
        let opts = SampleOptions {
            max_new: 150, // many times the window of 16: the cache is rebuilt again and again
            temperature: 0.9,
            top_k: 20,
            window: 16,
        };
        let tok = Tokenizer::Byte;
        let a = generate(
            &model,
            &tok,
            b"Ana",
            &opts,
            &mut Rng::new(8),
            &device,
            |_| {},
        );
        let b = generate(
            &model,
            &tok,
            b"Ana",
            &opts,
            &mut Rng::new(8),
            &device,
            |_| {},
        );
        assert_eq!(a.len(), 150);
        assert_eq!(a, b, "same seed, same text");
        // A prompt longer than the window and a zero-length request are fine too.
        let long_prompt = vec![b'x'; 100];
        let c = generate(
            &model,
            &tok,
            &long_prompt,
            &opts,
            &mut Rng::new(1),
            &device,
            |_| {},
        );
        assert_eq!(c.len(), 150);
        let none = SampleOptions { max_new: 0, ..opts };
        assert!(generate(&model, &tok, b"a", &none, &mut Rng::new(1), &device, |_| {}).is_empty());
    }

    #[test]
    fn greedy_generation_matches_a_step_by_step_full_pass() {
        use crate::train::{TrainOptions, train};
        // A briefly trained model has clearly separated logits, so rounding noise cannot flip
        // the greedy choice.
        let dir = std::env::temp_dir().join(format!("germen-gen-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus =
            crate::data::Corpus::from_bytes(crate::data::demo_corpus(40_000, 2).into_bytes(), 0.1)
                .unwrap();
        let cfg = Preset::Nano.config(64);
        let opts = TrainOptions {
            steps: 25,
            batch: 8,
            seq: 64,
            lr: 3e-3,
            warmup: 3,
            eval_every: 0,
            save_every: 0,
            ..TrainOptions::default()
        };
        let stop = std::sync::atomic::AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        let device = Device::flex();
        // The weights are read from the file lazily: keep the directory until the model is used.
        let (model, _, _) = crate::checkpoint::load(&dir, &device).unwrap();

        let tok = Tokenizer::Byte;
        let prompt = b"Ana cite";
        let greedy = SampleOptions {
            max_new: 30,
            temperature: 0.0,
            top_k: 1,
            window: 64,
        };
        let cached = generate(
            &model,
            &tok,
            prompt,
            &greedy,
            &mut Rng::new(1),
            &device,
            |_| {},
        );

        let mut context: Vec<TokenId> = tok.encode(prompt);
        assert_eq!(cached.len(), greedy.max_new);
        for (step, &want) in cached.iter().enumerate() {
            let t = context.len();
            let ids: Vec<i32> = context.iter().map(|&c| c as i32).collect();
            let x = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, t]), &device);
            let last: Vec<f32> = model
                .forward(x)
                .slice([0..1, t - 1..t, 0..256])
                .reshape([256])
                .into_data()
                .try_to_vec()
                .unwrap();
            let best = (0..256)
                .max_by(|&a, &b| last[a].total_cmp(&last[b]))
                .unwrap() as TokenId;
            assert_eq!(best, want, "token {step} differs");
            context.push(best);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_is_deterministic_for_a_fixed_seed() {
        let device = Device::flex();
        let model = Preset::Nano.config(16).init_seeded(&device, 1);
        let opts = SampleOptions {
            max_new: 8,
            ..SampleOptions::default()
        };
        let tok = Tokenizer::Byte;
        let a = generate(&model, &tok, b"", &opts, &mut Rng::new(42), &device, |_| {});
        let b = generate(&model, &tok, b"", &opts, &mut Rng::new(42), &device, |_| {});
        assert_eq!(a, b);
    }
}
