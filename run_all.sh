nosetests parsing/tests/test_upcfg.py  
nosetests parsing/tests/test_util.py
#Exercise 1
nosetests parsing/tests/test_baselines.py
python parsing/scripts/train.py -o rbranch -m rbranch 
python parsing/scripts/eval.py -i rbranch -m 20

python parsing/scripts/train.py -o lbranch -m lbranch 
python parsing/scripts/eval.py -i lbranch -m 20

python parsing/scripts/train.py -o flat -m flat
python parsing/scripts/eval.py -i flat -m 20

#Exercise 2
nosetests parsing/tests/test_cky_parser.py

#Exercise 3
python parsing/scripts/train.py -o upcfg -m upcfg 
python parsing/scripts/eval.py -i upcfg -m 20



#Exercise 4
for i in `seq 0 3`;
    do
        python parsing/scripts/train.py -o upcfg.trained.n$i -m upcfg:$i 
        python parsing/scripts/eval.py -i upcfg.trained.n$i -m 20
    done  
