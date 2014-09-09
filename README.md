Shogi Recognizer
================
Detect shogi (chess-like game) board state from photo.


Bootstrapping (planned)
----------
Written here for someone in the future, not programatically reproducible.
It's very hard and non-rewarding to maintain programs to insert manual
annotation (mostly rejection of failed samples) here and there.

1. Prepare photos with boards in the initial configuration `D0`
2. Extract piece patches from `D0` using (non-ML) image processing pipeline
3. Collect empty vs. non-empty labels using rotation-invariance, and train classifier `C0`
4. Use `C0` to `D0` and guess rotation
5. With rotations, we get labels for empty and non-promoted piece types, train classifier `C1` (also re-train `C0` -> `C0'`)
6. ???


Programs
----------
* `issue_id.py`: Merge files into a directory, assigning nice (safe, unique, short) keys
* `preprocess.py`: Extract cell images from raw photos
* `train.py`: Train cell image classifier


References
----------
* [J. C. Bazin, et al.: 3-line RANSAC for Orthogonal Vanishing Point Detection, IROS, 2012](http://graphics.ethz.ch/~jebazin/papers/IROS_2012.pdf)
