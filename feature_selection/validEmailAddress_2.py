"""
feature
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

valid email: score > 1
Accuracy of your model: 0.9615384615384616
"""

import numpy as np


WEIGHT = [                           
	[0.2],                              
	[0.5],
	[0.6],
	[-1],
	[-0.3],
	[-0.5],
	[-0.9],
	[0.7],
	[0.4],
	[-0.6]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	correct = 0
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		weight_vector = np.array(WEIGHT)
		weight_vector = weight_vector.T
		score = weight_vector.dot(feature_vector)
		print(score)
		if score > 1:
			correct += 1 if maybe_email in maybe_email_list[13:] else 0
		else:
			correct += 1 if maybe_email in maybe_email_list[:13] else 0
	accuracy = correct / len(maybe_email_list)
	print(accuracy)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with value 0's and 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
		if i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[0] and maybe_email.split('@')[1] else 0
		if i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.com' in maybe_email.split('@')[1] \
										or '.edu' in maybe_email.split('@')[1] \
										or '.org' in maybe_email.split('@')[1] else 0
		if i == 3:
			if feature_vector[1]:
				feature_vector[i] = 1 if maybe_email.split('@')[0][0] == '.'\
									or maybe_email.split('@')[0][-1] == '.' else 0
		if i == 4:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')) > 2 else 0
		if i == 5:
			if '"' in maybe_email and maybe_email.split('"')[0]:
				feature_vector[i] = 1 if maybe_email.split('"')[0][-1] != '.' else 0
		if i == 6:
			feature_vector[i] = 1 if '' in maybe_email.split('.') else 0
		if i == 7:
			if feature_vector[1]:
				for ele in maybe_email.split('@')[0]:
					if ele.isalpha():
						feature_vector[i] = 1
						break
		if i == 8:
			if feature_vector[1]:
				for ele in maybe_email.split('@')[0]:
					if ele.isdigit():
						feature_vector[i] = 1
						break
		if i == 9:
			if feature_vector[1]:
				for ele in maybe_email.split('@')[1]:
					if not ele.isalpha() and\
						ele != '.':
						feature_vector[i] = 1
						break
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that may be valid email addresses
	"""
	with open(DATA_FILE, 'r') as f:
		email_list = []
		for line in f:
			email_list.append(line)
	return email_list


if __name__ == '__main__':
	main()
