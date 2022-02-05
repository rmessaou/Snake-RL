# Snake-RL
A snake that knows how to win

# Reward
- eat: +10
- lose: -10
- else: 0

# Actions:
[1,0,0] -> straight
[0,1,0] -> turn right
[0,0,1] -> turn left

# State:

This is danger is close
- Straight Danger
- Right Danger
- Left Danger

Where is the snake facing
- Up Direction
- Down Direction
- Left Direction
- Right Direction

Where is the food wrt to the snake
- Up Food
- Down Food
- Right Food
- Left Food

# Model 
State (11) --> Hidden (?) --> Action (3)  

## Training
0. Init with some Q values
1. Predict Action (or Random for exploration)
2. Perform Action
3. Measure Reward
4. Update Q value and train with following params:
    1. NewQ(S,a) = Q(S,a) + alpha*[R + gamma*maxQ'(S',a')-Q(S,a)]
    2. loss = (NewQ(S,a)-Q(S,a))^2
