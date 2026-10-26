# Why not YOLO — and why the U-Net did not work either

This note records the reasoning behind the final architecture choice. The short
version: **cleaning is a pixel-reconstruction problem whose labels are defined by
a heuristic, so training a network to imitate that heuristic can only reproduce
its mistakes — and it cannot even do that when the input has already lost the
relevant information.**

## 1. Object detection is the wrong abstraction

YOLO answers the question *"where is the object, and which class is it?"*. Photon
cleaning asks *"what value should every pixel have?"*. Three concrete problems:

* **No object boundary.** A photon spot is a diffraction-limited blob whose
  intensity decays continuously into the background. Any bounding box is defined
  by an arbitrary threshold, and the box throws away the intensity profile that
  makes the photon measurable.
* **Detection is only half the task.** Even a perfect detector gives boxes and
  confidences. The image still has to be reconstructed: the detected photons must
  be re-synthesised at the right position with the right shape and the right
  intensity. That reconstruction is the hard part, and YOLO does not do it.
* **Annotation cost.** There are ~300 photon events per frame and 1133 frames.
  Drawing ~340 000 boxes by hand is not a realistic labelling budget, and the
  boxes would be inconsistent between annotators because the boundary is
  subjective.

A segmentation variant (YOLOv8-seg) fixes the boundary problem but keeps the
other two, and still optimises detection quality rather than reconstruction
quality.

## 2. The U-Net had the right shape but the wrong target

The obvious fix is to skip detection and learn the image-to-image mapping
directly. That is what the U-Net baseline did: 1133 pairs of
`input (noisy) -> target (cleaned)`, MSE loss plus a photon-conservation term.

It produced a smooth grey haze with a mean of ~9.3 ADU against a target mean of
~0.1 ADU, 99.95 % of pixels non-zero against 0.2–1.8 %, and SSIM ≈ 0.09
(`docs/results.md`, section 6). The failure has four distinct causes.

### 2.1 The target is not a function of the input patch

The `target` images were produced by the classical filter, which deletes any
bright pixel that is isolated. For a patch containing one bright isolated pixel
the target says **0** — regardless of whether that pixel was a hot pixel or a
real photon that landed alone. The network is asked to learn a mapping that is
not a function of the visible input; the best it can do is predict the
conditional mean of the target over all such patches, which is a faint grey
value. That is exactly what it produced.

This is not a capacity problem. It is an information problem: the answer is not
in the patch.

### 2.2 The loss rewards blur

With 0.2–1.8 % of pixels non-zero, mean squared error is dominated by the
background. Predicting a low constant everywhere is already a very good MSE
solution, and any sharp photon prediction that lands one pixel off is punished
quadratically. The photon-conservation term (`|Σpred − Σtrue|²`) constrains only
the total and can be satisfied by a uniform haze — which is what happened: the
prediction carries 230x the target energy, spread over the whole frame.

### 2.3 The preprocessing destroyed the signal

The dataset was built at 512 x 512 by resizing 1360 x 1040 frames with LANCZOS.
Measured on the shipped samples:

* photon events drop to **30 %** of their full-resolution count;
* the share of single-pixel events falls from **58 % to 13 %** — the very events
  the cleaning rule is about;
* the aspect ratio is distorted by **1.31x**, so a round photon becomes an
  ellipse.

By the time the network sees the data, five neighbouring photons are one blob
and the distinction between "isolated event" and "part of a cluster" — the core
of the task — is partly gone.

### 2.4 The dataset is smaller than it looks

1133 pairs sound like a lot, but they are consecutive frames of the same
illumination pattern from seven cameras. Neighbouring frames are almost
identical, so the effective number of independent scenes is closer to a few
dozen. A U-Net with ~2 M parameters will memorise those scenes rather than learn
a transferable rule.

## 3. What would make machine learning worthwhile

The failure is informative: it points at exactly which assumption to change.

1. **Train at full resolution.** Never downsample a signal whose information
   lives in single pixels.
2. **Predict the defect mask, not the clean image.** A binary "is this pixel a
   sensor defect" output is a well-posed classification problem, and the network
   can be evaluated with precision/recall instead of a global MSE.
3. **Get the labels from physics, not from a filter.** A defect recurs at the
   same coordinate; a photon does not. Feeding a short stack of frames and
   labelling pixels by temporal persistence gives labels that are *observable*
   rather than defined by the previous heuristic. This is the single change most
   likely to work.
4. **Then subtract.** Once the mask is reliable, cleaning is a subtraction, and
   the entire classical pipeline becomes a post-processing step with no learned
   reconstruction artefacts.

Until those labels exist, the classical pipeline is not a compromise — it is the
only version of the task with a well-defined answer.

## 4. Conclusion

| Approach | Verdict |
| --- | --- |
| YOLO / object detection | wrong abstraction: no object boundary, detection is not reconstruction, annotation cost is prohibitive |
| U-Net on downscaled pairs | failed for a measurable reason: unlearnable target, blur-friendly loss, destroyed input resolution, correlated dataset |
| Classical pipeline | works, is ~25–30 fps on one core, bit-exact reproducible, and its error modes are understood |
| Temporal defect map (next step) | the only ML formulation with physics-defined labels |

The practice report concluded that "U-Net is better suited than YOLO". The
stronger conclusion is that neither is the right frame for the problem as it was
posed, and that the label definition — not the architecture — is the thing to fix
first.
