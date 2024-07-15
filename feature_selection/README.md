# Feature Selection
select effective features and weights for a set of email data to increase recognition accuracy

## Data 
The first 13 entries in the file are invalid email addresses; the next 13 entries are valid email addresses

## Selected feature and weights in the first file
### feature
1. '@' in the str
2. No '.' before '@'
3. Some str before '@'
4. Some str after '@'
5. There is '.' after '@'
6. There is no white space
7. Ends with '.com'
8. Ends with '.edu'
9. Ends with '.tw'
10. Length > 10
### weights
1. 0.4
2. 0.4
3. 0.2
4. 0.2
5. 0.9
6. -0.65
7. 0.1
8. 0.1
9. 0.1
10. -0.7
### valid email: score > 0
### Accuracy of this model: 0.6538461538461539



## Selected feature and weights in the second file

### feature
1. '@' in the string
2. there is string before and after '@'
3. there is '.com' or '.edu' or '.org' after '@'
4. start with or end with '.' before '@'
5. more than one '@'
6. no '.' before "" if there is ""
7. nothing between two '.'
8. there is alpha before '@'
9. there is digit before '@'
10. something other than alpha or '.' after '@'

### weights
1. 0.2
2. 0.5
3. 0.6
4. -1
5. -0.3
6. -0.5
7. -0.9
8. 0.7
9. 0.4
10. -0.6
### valid email: score > 1
### Accuracy of model: 0.9615384615384616
