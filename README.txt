Exercise 1:

sents count: 17379
word vocabulary size: 46482
tag vocabulary size: 48

most frequent tags:
==================
[('nc', 92002), ('sp', 79904), ('da', 54552), ('vm', 50609), ('aq', 33904), ('fc', 30148), ('np', 29113), ('fp', 21157), ('rg', 15333), ('cc', 15023)]

% of total for most frequent tags: [('nc', 92002, 17.786137940100684), ('sp', 79904, 15.447311645027337), ('da', 54552, 10.546177223412236), ('vm', 50609, 9.783903121785999), ('aq', 33904, 6.554435998360618), ('fc', 30148, 5.828313369471918), ('np', 29113, 5.628223667421917), ('fp', 21157, 4.0901428273158205), ('rg', 15333, 2.964227441094365), ('cc', 15023, 2.9042971921711764)]

Top words in most frequent tags:
================================
nc [('años', 849), ('presidente', 682), ('millones', 616), ('equipo', 457), ('partido', 438)]
sp [('de', 28475), ('en', 12114), ('a', 8192), ('del', 6518), ('con', 4150)]
da [('la', 17897), ('el', 14524), ('los', 7758), ('las', 4882), ('El', 2817)]
vm [('está', 564), ('tiene', 511), ('dijo', 499), ('puede', 381), ('hace', 350)]
aq [('pasado', 393), ('gran', 275), ('mayor', 248), ('nuevo', 234), ('próximo', 213)]
fc [(',', 30148)]
np [('Gobierno', 554), ('España', 380), ('PP', 234), ('Barcelona', 232), ('Madrid', 196)]
fp [('.', 17513), ('(', 1823), (')', 1821)]
rg [('más', 1707), ('hoy', 772), ('también', 683), ('ayer', 593), ('ya', 544)]
cc [('y', 11211), ('pero', 938), ('o', 895), ('Pero', 323), ('e', 310)]

Ambiguity levels
================
level:  2
# of words:  2194 0.047201067079729785
top words in level [('la', 18100), ('y', 11212), ('"', 9296), ('los', 7824), ('del', 6519)]
level:  3
# of words:  153 0.0032915967471279207
top words in level [('.', 17520), ('a', 8200), ('un', 5198), ('no', 3300), ('es', 2315)]
level:  4
# of words:  19 0.0004087603803622908
top words in level [('de', 28478), ('dos', 917), ('este', 830), ('tres', 425), ('todo', 393)]
level:  5
# of words:  4 8.6054816918377e-05
top words in level [('que', 15391), ('mismo', 247), ('cinco', 224), ('medio', 105)]
level:  6
# of words:  3 6.454111268878275e-05
top words in level [('una', 3852), ('como', 1736), ('uno', 335)]
level:  7
# of words:  0 0.0
top words in level []
level:  8
# of words:  0 0.0
top words in level []
level:  9

Exercise 3:

Accuracy: 89.03%
Accuracy known: 95.35%
Accuracy unknown: 31.80%

Check the matrix plot in tagging/baseline_confusion.png

Exercise 5:
N	Accuracy,		known,		unknown
2 	92.33%			97.40%		46.36%	
3  93.05% 			97.47% 		53.06%
4  92.18%         96.43%      53.66%



Exercise 7:

N	Accuracy,		known,		unknown
1	89.22%			93.43%		51.13%
2	87.38%			90.78%		56.53%
3  88.44%         91.99%		56.31%
4  88.76%         92.16%		57.94%	

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


N	Accuracy,		known,		unknown
1	89.79%			93.67%		54.58%
2	88.26% 			91.47%		59.15%
3	88.84%			92.23%		58.17%
4	89.01%			92.26%		59.60%


Next come the results for including an extra feature that will trigger when the word has one of the following suffixes, {'ando', 'ado', 'ar', 'er', 'ir'}.
This is an effort to generalize better over unknown words.

N	Accuracy,		known,		unknown
1	90.16%			93.98%		55.53%
2	88.97%			92.10%		60.59%
3  89.28%         92.70%		58.26%
4  89.46%         92.57%		61.21%	         	


With all these features the results for SVM,
N	Accuracy,		known,		unknown
1 	92.36%			96.40%		55.77%
2 	92.43%			95.83%		61.54%
3 	93.41%			96.91%		61.63%
4 	93.53%			96.91%		62.92%

and for MultinomialNB,
N	Accuracy,		known,		unknown
1 	87.11%			91.44%		47.89%
2 	81.95%  		85.13%		53.13%	
3	82.96% 			86.55% 		50.50%
4 	82.01% 			85.69% 		48.62%

