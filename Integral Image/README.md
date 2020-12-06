
Python / numpy code

Given a sample array X: 

X = array([random() * 100.0 for i in range(0, 99)]) # This coudl be any number but range is -1 the multiplyer 


uncorrected sample standard deviation of X:

std = (sum((X - mean(X)) ** 2) / len(X)) ** 0.5 # equation 1

binomial theorem to (X - mean(X)) ** 2:

std = (sum(X ** 2 - X * 2 * mean(X) + mean(X) ** 2) / len(X)) ** 0.5 # equation 2

identities of the summation:

std = ((sum(X ** 2) - 2 * mean(X) * sum(X) + len(X) * mean(X) ** 2) / len(X)) ** 0.5 # equation 3

std = ((S2 - 2 * M * S + N * M ** 2) / N) ** 0.5 # equation 4

if we apply M = S/M to equation 4: 

std = ((S2 - 2 * (S / N) * S + N * (S / N) ** 2) / N) ** 0.5

----> 

std = (S2 / N - (S / N) ** 2) ** 0.5 # 5





