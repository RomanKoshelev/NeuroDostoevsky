import numpy as np

class MathGenerator:
    def __init__(self, first, last):
        self._first = first
        self._last = last

    def _get_rand_nums(self, sign):
        if sign == '=':
            n1 = np.random.randint(self._first, self._last)
            n2 = n1
        else:
            raise NotImplementedError
        return n1,n2

    def _sort_exp(self, exp):
        exp = np.copy(exp)
        np.random.shuffle(exp)
        return list(exp)
        
    def _to_sum_list(self, n):
        if n==0:
            return[]
        if n<=1:
            return[n]
        a = np.random.randint(1,n+1)
        return [a] + self._to_sum_list(n-a)

    @staticmethod
    def _to_sent(exp1, sign, exp2):
        sent1 = '+'.join(['%s' %n for n in exp1])
        sent2 = '+'.join(['%s' %n for n in exp2])
        return '%s %s %s' % (sent1, sign, sent2)

    def _generate_sent(self, sign):
        num1,num2 = self._get_rand_nums(sign)
        exp1 = self._to_sum_list(num1)
        exp2 = self._to_sum_list(num2)
        exp1 = self._sort_exp(exp1)
        exp2 = self._sort_exp(exp2)
        if sign == '=':
            assert num1 == np.sum(exp1)
            assert num2 == np.sum(exp2)
            assert num2 == num1
        return self._to_sent(exp1, sign, exp2)

    def generate(self, signs, per_sign):
        sents = []
        for sign in signs:
            for i in range(per_sign):
                sents.append(self._generate_sent(sign))
        return sents