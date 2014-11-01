Shogi Recognizer
================
Detect shogi (chess-like game) board state from photo.

Usage
----------
Run `./derive_nets.py`.

See `./analyze.py` for further usage.


Training
----------
Use `./manage_dataset.py` to import images.
Run `./annotate_dataset.py` and its web UI to annotate images.


Run `./derive_nets.py`.
`$(CAFFE_ROOT)/build/tools/caffe train -solver cells-plan.prototxt`

Move the final `cells_iter_*.caffemodel` to `params/cells.caffemodel`

Now it's ready!


Programs
----------
* `analyze.py`: Analyze given photo (from CLI or python interface)
* `issue_id.py`: Merge files into a directory, assigning nice (safe, unique, short) keys
* `preprocess.py`: Extract cell images from raw photos
* `classify.py`: Train / use cell image classifier


Dependency
----------
You need [lsd-python](https://github.com/xanxys/lsd-python) line segment
detector to run `./preprocess.py`. Note that lsd-python is Affero GPL3,
while this repo is MIT licensed (i.e. You can't use this python code
on server side without disclosing server source code).


Useful Techniques
----------
### Manually selecting good data generated from unstable detector
`./preprocess.py --debug` forces fixed seed, with that, you can update `--blacklist` to reject bad data after inspecting data.


Recognition Steps
----------
![9x9 Grid Detection](doc/validness.jpeg)


References
----------
* [J. C. Bazin, et al.: 3-line RANSAC for Orthogonal Vanishing Point Detection, IROS, 2012](http://graphics.ethz.ch/~jebazin/papers/IROS_2012.pdf)
