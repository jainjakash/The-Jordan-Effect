# The-Jordan-Effect

Brief description of the Jordan Project: I built a multilayered neural network that took in relevant statistical information about Michael Jordan’s performance(age, home vs. away, points, rebounds, assists, and age) in order to predict whether or not the Bulls won the game. The training data was amassed from online datasets, which I parsed and normalized. 

In my writeup of the project, I interestingly concluded that the least relevant factors were assists and rebounds, which seemed odd at first for a player that nearly averaged a triple double for his career. It was only upon further analysis that I realized that the Bulls’s most successful seasons came when Pippen had a majority of the assists and Rodman had a majority of the rebounds.

Setback I faced: Initially, the neural network relied too heavily on age and points and virtually nothing else. With careful debugging, I realized this was because these categories were artificially higher in value. By scaling each of the categories to a number between 0 and 1, the network’s accuracy with sample data jumped from 65% to 95%. 
