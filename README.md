Fake news on social media is often multimodal, and
real posts do not always have both text and image available. We
build a distributed PySpark pipeline for the Fakeddit dataset
(fine-grained, 6-way classification) that performs ETL, GPU-
batched feature extraction, and MLlib/PyTorch classification, and
use it to compare a concatenation baseline against several adap-
tive, gated fusion designs. A confidence-weighted gate trained
separately from the classifier does not beat simple concatenation
– we trace this to the gate never being optimized against the
downstream loss. Training the gate jointly with the classifier,
end-to-end, fixes this: our final model, adaptive-joint, beats
the concatenation baseline in all three tested conditions (both
modalities present, image missing, text missing), with the largest
gain on clean data (65.1% vs. 56.2% accuracy
