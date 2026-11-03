# Decoupled Cost-Sensitive Hard Router (DLSR)

This isolated experiment freezes the current five-class base and specialists,
then trains a lightweight global/local router from frozen base features. Model
and hard-policy selection use five-fold CHIME development-only OOF predictions.
Paper-test is evaluated only after the configuration is locked.
