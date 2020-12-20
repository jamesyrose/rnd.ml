
# SpaceGroup Prediction
### Using random forest regression for space group prediction from XRD 

Basic Random forest achieves ~51% on barvais  lattice and 47% on space groups

Space group 27 and 82 have highest error rate consisting of approximately 9% of the error. Stacking with a binary classifier could reduce that error. Similarly, 7, 34, and 67 have fairly high error rates, another classifier for it could reduce error. 

Binary classifier would be along the lines of it is this space group or not. For example, for Space group 27, label 27 as 1 and the rest as 0. 

[This paper](https://www.sciencedirect.com/science/article/pii/S2666389920300131#bib38) achieves up to 85% accuracy.
