



Exercise 7:

N	Accuracy,		known,		unknown
1	89.22%			93.43%		51.13%
2	87.38%			90.78%		56.53%
3   88.44%          91.99%		56.31%
4   88.76%		    92.16%		57.94%	

For N=3, 	aq -> nc ~ 1186 times
         	nc -> aq ~ 902 times
         	nc -> np ~ 1208 times
         	aq -> np ~ 666 times
         	rg -> np ~ 218 times
         	rg -> vm ~ 260 times
         	nc -> vm ~ 460 times

For the first and second lines we have things like "chico" that can be both an adjective and a noun.

New features:
aq -> np : modify is_title so it doesn't trigger at the beginning of sentence. Aqs can be confused with nps if you capitalize them at the beginning of sentence.
Add another feature to specify beginning of sentence.
aq -> nc :  see if a verb has already appeared before. E.g., when tagging "El hombre es chico",
most likely chico will be an aq. Look into a window of say 2 or 3 before current word.
rg -> np/vm: detect the "mente" suffix. E.g., "timidamente"


      	
For N=3, 	aq -> nc ~ 1102 times +
         	nc -> aq ~ 926 times -
         	nc -> np ~ 1443 times -
         	aq -> np ~ 736 times -
         	rg -> np ~ 178 times +
         	rg -> vm ~ 151 times +
         	nc -> vm ~ 500 times -     	

Exercise 7:

N	Accuracy,		known,		unknown
1	89.79%			93.67%		54.58%
2	88.26% 			91.47%		59.15%
3	88.84%			92.23%		58.17%
4	89.01%			92.26%		59.60%


Next come the results for including an extra feature that will trigger when the word has one of the following suffixes, {}

N	Accuracy,		known,		unknown
1	90.16%			93.98%		55.53%
2	88.97%			92.10%		60.59%
3   89.28%          92.70%		58.26%
4   89.46%		    92.57%		61.21%	         	



