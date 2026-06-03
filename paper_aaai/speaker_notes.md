# Speaker Notes — Fairds Talk

> **Goal: about 8 minutes** (safe under the 9-minute limit).
> Easy English. Short sentences. Speak **slowly** and look at the audience.
> On the chart slides (6 and 7), point at the bars while you talk.
> If you run long, the slides you can shorten are 9 and 10.

**Time plan (total ≈ 8:10)**

| Slide | Topic | Time | Clock |
|---|---|---|---|
| 1 | Title | 0:25 | 0:00–0:25 |
| 2 | Problem | 0:50 | 0:25–1:15 |
| 3 | Idea | 0:45 | 1:15–2:00 |
| 4 | Method | 0:55 | 2:00–2:55 |
| 5 | Is it real? | 0:40 | 2:55–3:35 |
| 6 | Residual test | 0:55 | 3:35–4:30 |
| 7 | Main result | 0:50 | 4:30–5:20 |
| 8 | STL-10 numbers | 0:50 | 5:20–6:10 |
| 9 | Honest scope | 0:45 | 6:10–6:55 |
| 10 | Cost | 0:35 | 6:55–7:30 |
| 11 | Wrap-up | 0:40 | 7:30–8:10 |

---

## Slide 1 — Title  ·  (0:00–0:25)

> Hello everyone. Today I will show a simple way to make a model more robust.
> The problem is data shortcuts. Our method is called Fairds.
> It uses a data-value score during training.
> We test it on three different image tasks.

---

## Slide 2 — The problem  ·  (0:25–1:15)

> Let me explain the problem. A model often learns an easy shortcut.
> For example, it may look at the image background, not the real object.
> The shortcut works during training. But at test time, the shortcut flips.
> So the model fails on one group of the data. We call it the **worst group**.
> The average accuracy still looks good. That is the danger.
> Old fixes need group labels, or a second training stage, or a slow inner loop.
> We want something simpler. So we ask one question.
> Can a closed-form score, computed in one pass, fix this? No group labels. No extra loop.

---

## Slide 3 — The idea  ·  (1:15–2:00)

> Here is our idea. There is a tool called In-Run Data Shapley.
> It gives each training sample a score.
> The score says how much that sample helps a small, balanced check set.
> The nice part is: it is closed-form. We get it during training, with no retraining.
> We turn this score into a weight, and we do weighted training.
> A sample with a higher score gets more weight.
> So the whole question becomes: what score should we use?

---

## Slide 4 — Method  ·  (2:00–2:55)

> This slide shows the score. The first-order score is simple.
> It is the inner product of two gradients: one from the sample, one from the check set.
> We compute all the sample gradients in one pass.
> Now the main part. We add a **second-order** term. It uses the curvature, the Hessian.
> We get it with the Pearlmutter trick. It needs only one more backward pass.
> One problem: the curvature size changes a lot during training.
> So we **rescale** the new term to the same size as the first term, every minibatch.
> This step is important. Without it, the method breaks.

---

## Slide 5 — Is the second-order term real?  ·  (2:55–3:35)

> Now a fair question. Maybe this second-order term is just smoothing. Maybe it is only noise.
> We checked this, and we found something interesting.
> The check-set gradient is almost the top direction of the curvature.
> So the new term is almost parallel to the first score.
> We split it into two parts: one parallel part, which is like smoothing,
> and one extra part — the **residual** — that points a different way.
> So the real question is: is this residual a true signal, or just noise?

---

## Slide 6 — Residual test  ·  (3:35–4:30)

> To answer this, we ran a clean test. We made five versions.
> They all share the first-order score. They only treat the residual differently.
> One uses the real residual. One shuffles it. One keeps only the parallel part. One flips its sign.
> Look at the bars. Real is best. Shuffle is worse. Parallel is worse still.
> And flipping the sign breaks everything.
> On CIFAR, real beats shuffle by about six points, with a small p-value.
> If the residual were just noise, shuffling it would not matter. But it does matter.
> So the residual is a real, useful signal.

---

## Slide 7 — Main result  ·  (4:30–5:20)

> This is our main result. We test the same method on three problems.
> A color shortcut on MNIST. A texture shortcut on CIFAR. And natural photos on STL-10.
> The grey and blue bars use **no** group labels.
> The red bars use group labels or a two-stage trick.
> Now look at our method — the striped bar.
> It beats every no-label baseline, in all three problems.
> So this is not one lucky result. The same idea works for color, texture, and real photos.

---

## Slide 8 — STL-10 numbers  ·  (5:20–6:10)

> Here are the numbers for the hardest task, STL-10, with real photos.
> Our method reaches thirty-two percent worst-group accuracy. Plain training is near zero.
> We beat plain training by twenty-eight points.
> We beat the bi-level baseline by seventeen points.
> And, very important, we beat the first-order score by thirteen points.
> That last gap is the second-order effect, alone.
> It is the same effect we saw on CIFAR, now on a third, harder task.
> All the p-values are small, over ten seeds.

---

## Slide 9 — Honest scope  ·  (6:10–6:55)

> Now let us be honest. We do not claim the best accuracy.
> Two-stage methods like JTT, and group-label methods like GroupDRO, are stronger.
> Our value is the simple, closed-form mechanism.
> And there is a case where our method fails.
> On Waterbirds, with a pretrained network, it does not match the bi-level baseline.
> The reason is clear. With a strong pretrained model, the early gradients are small and noisy.
> So our score buffer fills with noise. In that case, use bi-level instead.
> We report this clearly.

---

## Slide 10 — Cost  ·  (6:55–7:30)

> A quick word on cost.
> The first-order version is about four times slower than plain training.
> That is cheaper than the bi-level baseline.
> The second-order version is about seven to eight times slower, because of the Hessian step.
> So use the second-order term when the residual signal matters —
> the from-scratch tasks we just showed.

---

## Slide 11 — Wrap-up  ·  (7:30–8:10)

> To wrap up. We turn a data-value score into a training reweighter.
> We add a second-order term with rescaling.
> With the ablation, we prove the gain is the residual, not just smoothing.
> And the mechanism works across three problems.
> We are also honest about where it fails.
> The big message: a closed-form data score can shape training, inside a clear regime.
> Thank you. I am happy to take any questions.

---

### Quick tips
- **Practice once with a timer.** If you finish slide 7 by 5:20, you are on track.
- Breathe between slides. A short pause is fine.
- You do not need to read every word — these notes are a guide, not a script to memorize.
- If a question comes early, it is okay; you have an ~0.8-minute buffer.
