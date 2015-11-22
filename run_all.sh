
nosetests parsing/tests/test_baselines.py  
nosetests parsing/tests/test_cky_parser.py
nosetests parsing/tests/test_upcfg.py  
nosetests parsing/tests/test_util.py

#Exercise 4
for i in `seq 0 3`;
    do
        python parsing/scripts/train.py -o upcfg.trained.n$i -m upcfg:$i 
        python parsing/scripts/eval.py -i upcfg.trained.n$i -m 20
    done  