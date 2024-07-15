"""
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

feature
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

valid email: score > 0
Accuracy of this model: 0.6538461538461539
"""

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	correct = 0
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		score = sum(feature_vector[i] * WEIGHT[i][0] for i in range(len(feature_vector)))
		if score > 0:
			correct += 1 if maybe_email in maybe_email_list[13:] else 0
		else:
			correct += 1 if maybe_email in maybe_email_list[:13] else 0
	accuracy = correct/len(maybe_email_list)
	print(accuracy)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[0] else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[1] else 0
		elif i == 4:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' in maybe_email[maybe_email.find('@'):] else 0
		elif i == 5:
			feature_vector[i] = 1 if ' ' not in maybe_email else 0
		elif i == 6:
			feature_vector[i] = 1 if maybe_email[-4:] == '.com' else 0
		elif i == 7:
			feature_vector[i] = 1 if maybe_email[-4:] == '.edu' else 0
		elif i == 8:
			feature_vector[i] = 1 if maybe_email[-3:] == '.tw' else 0
		elif i == 9:
			feature_vector[i] = 1 if len(maybe_email) > 10 else 0
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	with open(DATA_FILE, 'r') as f:
		email_list = []
		for line in f:
			email_list.append(line)
	return email_list


if __name__ == '__main__':
	main()
