# Deep Fizz Buzz

Uses TensorFlow to predict the first 100 values of [Fizz buzz](https://en.wikipedia.org/wiki/Fizz_buzz).

## Installation

```bash
$ pip install -r requirements.txt
```

If you want to run this application with GPU accelleration please see the installation notes in my [TensorFlow on a GTX 1080](http://tech.marksblogg.com/tensorflow-nvidia-gtx-1080.html) blog post, otherwise you can use the CPU-accellerated version of Tensorflow below.

```bash
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
```

## Running

```
$ python fizz_buzz.py
```

The results will look something like the following:

```
6 wrong value(s) (position, correct, predicted):
20 fizz 21
41 fizz 42
68 fizz buzz
80 fizz 81
86 fizz 87
92 fizz 93

Predictions TensorFlow made:
['1' '2' 'fizz' '4' 'buzz' 'fizz' '7' '8' 'fizz' 'buzz' '11' 'fizz' '13'
 '14' 'fizzbuzz' '16' '17' 'fizz' '19' 'buzz' '21' '22' '23' 'fizz' 'buzz'
 '26' 'fizz' '28' '29' 'fizzbuzz' '31' '32' 'fizz' '34' 'buzz' 'fizz' '37'
 '38' 'fizz' 'buzz' '41' '42' '43' '44' 'fizzbuzz' '46' '47' 'fizz' '49'
 'buzz' 'fizz' '52' '53' 'fizz' 'buzz' '56' 'fizz' '58' '59' 'fizzbuzz'
 '61' '62' 'fizz' '64' 'buzz' 'fizz' '67' '68' 'buzz' 'buzz' '71' 'fizz'
 '73' '74' 'fizzbuzz' '76' '77' 'fizz' '79' 'buzz' '81' '82' '83' 'fizz'
 'buzz' '86' '87' '88' '89' 'fizzbuzz' '91' '92' '93' '94' 'buzz' 'fizz'
 '97' '98' 'fizz' 'buzz']
```

There are various options you can pass for adjusting the training parameters:

```
Usage: fizz_buzz.py [OPTIONS]

Options:
  --digits INTEGER
  --hidden_units INTEGER
  --learning_rate FLOAT
  --iterations INTEGER
  --help                  Show this message and exit.
```
