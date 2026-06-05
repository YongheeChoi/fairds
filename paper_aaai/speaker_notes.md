# Speaker Notes — Fairds Talk (Group Fairness)

> **Goal: about 8 minutes** (safe under the 9-minute limit).
> Easy English. Short sentences. Speak **slowly** and look at the audience.
> On the chart slides (6, 7, 8), point at the bars while you talk.
> If you run long, the slides you can shorten are 9 and 10.

**Time plan (total ≈ 8:10)**

| Slide | Topic | Time | Clock |
|---|---|---|---|
| 1 | Title | 0:25 | 0:00–0:25 |
| 2 | Problem: bias → unfairness | 0:50 | 0:25–1:15 |
| 3 | Idea | 0:45 | 1:15–2:00 |
| 4 | Method | 0:55 | 2:00–2:55 |
| 5 | Is it real? | 0:40 | 2:55–3:35 |
| 6 | Residual test | 0:55 | 3:35–4:30 |
| 7 | Group fairness, 3 regimes | 0:50 | 4:30–5:20 |
| 8 | Closing the gap | 0:50 | 5:20–6:10 |
| 9 | Demographic (honest) | 0:45 | 6:10–6:55 |
| 10 | Cost | 0:35 | 6:55–7:30 |
| 11 | Wrap-up | 0:40 | 7:30–8:10 |

---

## Slide 1 — Title  ·  (0:00–0:25)

> Hello everyone. This work is about fairness.
> A model often copies the bias in its data.
> We use a simple data-value score to stop that, during training, with no group labels.
> We show it on three kinds of bias: color, texture, and natural images.

---

## Slide 2 — Bias becomes unfairness  ·  (0:25–1:15)

> Here is the problem. In the training data, a majority group ties some attribute to the label.
> For example, a background, a color, or a demographic feature.
> An ordinary model just uses that biased attribute.
> The average accuracy looks good. But the model fails on the minority group,
> where the attribute does not hold.
> That is an unfair model: good for the majority, poor for the minority.
> The strong fixes need extra help: group labels, a second training stage, or a bi-level loop.
> We ask a simpler question.
> Can a closed-form, one-pass score, with no group labels, find the examples that teach
> the bias — so we can down-weight them?

---

## Slide 3 — The idea  ·  (1:15–2:00)

> Our idea uses In-Run Data Shapley.
> It gives each training example a score: how much it helps a small, group-balanced reference set.
> This is closed-form, computed during training.
> Here is the key point. An example that helps the biased majority, but not the balanced set,
> gets a low score.
> We turn the score into a weight. A low score means less weight.
> So the question becomes: what score best targets the bias?

---

## Slide 4 — Method  ·  (2:00–2:55)

> This is the score. The first-order score is the inner product of two gradients —
> one from the example, one from the balanced set. We compute them in one pass.
> Then we add a second-order term, using the curvature, with the Pearlmutter trick.
> It needs just one more backward pass.
> The key trick: we rescale this term to the same size as the first term, every minibatch,
> because the curvature size changes a lot during training.
> Without this rescaling, the method breaks under strong bias.

---

## Slide 5 — Is the second-order term real?  ·  (2:55–3:35)

> A fair worry: maybe the second-order term is just smoothing, not a real bias signal.
> We checked.
> The balanced-set gradient is almost the top direction of the curvature.
> So the new term is almost parallel to the first score.
> We split it into a parallel part, which is just smoothing,
> and an extra part — the residual.
> So the real question is: is the residual a true bias signal, or only noise?

---

## Slide 6 — Residual test  ·  (3:35–4:30)

> We tested this with five versions.
> They all share the first score, but treat the residual differently:
> real, shuffled, parallel-only, and sign-flipped.
> The order is clear. Real is best, shuffle is worse, parallel is worse still,
> and flipping the sign breaks everything.
> On CIFAR, the real residual beats the shuffled one by about six points.
> If the residual were just noise, shuffling it would not matter. But it does.
> So the bias-correcting signal is real.

---

## Slide 7 — Group fairness across three regimes  ·  (4:30–5:20)

> This is our main fairness result.
> We test three kinds of bias: color, texture, and natural images.
> The bars show worst-group accuracy, which protects the minority group.
> Our method — the striped bar — beats every no-group-label baseline, in all three.
> So the same idea works across very different biases.
> The red bars, group-label and two-stage methods, are fairer still — but they use more help.

---

## Slide 8 — Closing the gap by helping the minority  ·  (5:20–6:10)

> We can also measure fairness as the accuracy gap between the majority and the minority group.
> Our method cuts that gap by forty to forty-seven percent over plain training.
> And, very important, it closes the gap by helping the minority, not by hurting the majority.
> On STL-10, minority accuracy goes from four percent to thirty-two,
> while the majority drops only a little, and overall accuracy still goes up.
> This is what fairness should do: protect the under-represented group.
> To be honest, JTT and GroupDRO are fairer, but they use a second stage or group labels.

---

## Slide 9 — Demographic fairness: structured vs diffuse  ·  (6:10–6:55)

> Now, does this help real demographic data? It depends on the bias.
> On CelebA, we predict blond hair, with gender as the sensitive group. This is a structured image bias.
> Here our method works strongly: worst-group accuracy goes from twenty percent to seventy-two percent,
> and the equal-odds gap drops by more than half. Accuracy even goes up.
> But on tabular data, Adult and COMPAS, the bias is diffuse — spread over many weak features.
> There our method keeps accuracy but barely moves the gaps.
> So the scope is clear: structured bias works, even on real faces and real demographics;
> diffuse tabular bias does not.

---

## Slide 10 — Cost  ·  (6:55–7:30)

> A quick word on cost.
> The first-order version is about four times slower than plain training —
> that is cheaper than the bi-level method.
> The second-order version is about seven to eight times slower, because of the curvature step.
> Use it for structured bias, with no group labels, when you must keep accuracy.

---

## Slide 11 — Wrap-up  ·  (7:30–8:10)

> To wrap up. We turn a data-value score into a fair-training reweighter.
> We add a second-order term that targets the bias.
> We prove, with the ablation, that the signal is real, not just smoothing.
> And it improves group fairness across three kinds of bias,
> while we stay honest about the demographic boundary.
> The big message: a closed-form data-value score can keep bias out of a model as it trains,
> within a clear regime.
> Thank you. I am happy to take any questions.

---

### Quick tips
- **Practice once with a timer.** If you finish slide 7 by 5:20, you are on track.
- Breathe between slides. A short pause is fine.
- These notes are a guide — you do not need to read every word.
- Key fairness words to say clearly: *minority group*, *accuracy gap*, *no group labels*, *structured bias*.
