[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AiU9r98G)
# Spring 26 CSE 473/573 Project 2

## TODO: Fill the blank with your UBIT name

Please update this README.md to fill the blank below before submiting to UB Learn!

Name: __________

UBIT: __________

## Implementation

The following commands should be executed in the root folder of the project.

### Task 1

```bash
python task1.py --input_path images/t1 --output outputs/task1.png
```

### Task 2

```bash
python task2.py --input_path images/t2 --output outputs/task2.png --json task2.json
```

### Bonus
We have provided some challenging examples in the `images/Bonus1` folder — you’re encouraged to see if you can create a panorama from them.

You can also add your own Buffalo- or UB-related images to the `images/Bonus2` folder.

```bash
# Bonus1
python task2.py --input_path images/Bonus1 --output outputs/bonus1.png --json bonus1.json

# Bonus2
python task2.py --input_path images/Bonus2 --output outputs/bonus2.png --json bonus2.json
```

**Note**: In the commands, use `python3` if your environment has python named as `python3` instead `python`.

## Pack your submission

Note that when packing your submission, the script would run your code before packing.

```bash
bash pack_submission.sh <Your_UBIT_name>
```
Change **<Your_UBIT_name>** with your UBIT name.
The resulting zip file should be named **"submission\_<Your_UBIT_name>.zip"**, and it should contain 2 files, named **"task2.json"**, **"stitching.py"** and **images** folder and **outputs** folder, and optional **"bonus1.json"** and **"bonus2.json"** files.  If not, there is something wrong with your code/filename, please go back and check.

Submision:
1. You should only submit the zip file to UBLearns - Assignments - Project2 (This file only will be graded)
2. You should also push your latest code to your classroom repository. (This file is only for progress tracking and can influence your project grade based of the .zip submission on the UBLearns)
