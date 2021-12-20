#!/bin/bash

mkdir -p checkpoints/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-Bfq72aa6ZtOt5rEsUJBHoDRT2X6YovL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Bfq72aa6ZtOt5rEsUJBHoDRT2X6YovL" -O "./checkpoints/gen1.pth" && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-jUnywvuAc-qJ0nUKo6mPteYIbIiDiZM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-jUnywvuAc-qJ0nUKo6mPteYIbIiDiZM" -O "./checkpoints/gen2.pth" && rm -rf /tmp/cookies.txt

mkdir ./test_data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C1gxDJSl5Ho3Z2-rLF8GbpEAimPPajRk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C1gxDJSl5Ho3Z2-rLF8GbpEAimPPajRk" -O "./test_data/test_data.zip" && rm -rf /tmp/cookies.txt
unzip ./test_data/test_data.zip -d ./

python test.py -m ./checkpoints/gen1.pth -d ./test_data/ -o ./test_1/
python test.py -m ./checkpoints/gen2.pth -d ./test_data/ -o ./test_2/
