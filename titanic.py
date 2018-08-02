import csv
import numpy as np
import random

TRAIN_CSV_PATH = "/Users/Lin/Documents/workspace/pythonArea/Titanic/train.csv"
TEST_CSV_PATH = "/Users/Lin/Documents/workspace/pythonArea/Titanic/test.csv"


class Titanic:
    def __init__(self):
        self.train_psg = []
        self.train_label = []
        self.test_psg = []
        self.impute_train = []
        self.impute_test = []
        self._load_csv()
        self._preprocess()

    def _load_csv(self):
        def check_train(s):
            for idx in range(8):
                if len(s[idx]) == 0 and idx != 3:
                    return False
            return True
        
        def check_test(s):
            for idx in range(7):
                if len(s[idx]) == 0 and idx != 2:
                    return False
            return True

        train_cnt = 0
        skipped_cnt = 0
        with open(TRAIN_CSV_PATH, 'rt') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                if not train_cnt:
                    train_cnt += 1
                    continue
                if not check_train(row):
                    self.impute_train.append(row[2:8])
                    train_cnt += 1
                    skipped_cnt += 1
                    continue
                survive = int(row[1])
                # pclass = float(row[2])
                # sex = 0 if row[4] == 'male' else 1
                # age = float(row[5])
                # sibsp = float(row[6])
                # parch = float(row[7])
                self.train_psg.append(self._make_psg(row[2:]).get_info())
                self.train_label.append(survive)
                train_cnt += 1
        print("Total train data: {}, skipped: {}".format(train_cnt, skipped_cnt))
        test_cnt = 0
        skipped_cnt = 0
        with open(TEST_CSV_PATH, 'rt') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                if not test_cnt:
                    test_cnt += 1
                    continue
                if not check_test(row):
                    self.impute_test.append(row[1:7])
                    train_cnt += 1
                    skipped_cnt += 1
                    continue
                # pclass = float(row[1])
                # sex = 0 if row[3] == 'male' else 1
                # age = float(row[4])
                # sibsp = float(row[5])
                # parch = float(row[6])
                self.test_psg.append(self._make_psg(row[1:]).get_info())
                test_cnt += 1
        print("Total test data: {}, skipped: {}".format(test_cnt, skipped_cnt))

    def _make_psg(self, s):
        pclass = float(s[0])
        sex = 0 if s[2] == 'male' else 1
        age = float(s[3])
        sibsp = float(s[4])
        parch = float(s[5])
        return Passenger(pclass, sex, age, sibsp, parch)
    
    def _preprocess(self):
        self.train_data = np.array([np.array(i) for i in self.train_psg])
        self.test_data = np.array([np.array(i) for i in self.test_psg])
        self.train_data = self.train_data.astype(float)
        self.test_data = self.test_data.astype(float)
        t_min, t_max, t_mean = np.min(self.train_data, axis=0), np.max(self.train_data, axis=0), np.mean(self.train_data, axis=0)
        self._impute(t_mean)
        self._impute(t_mean, is_test=True)
        self.train_data = (self.train_data - (t_max + t_min)/2) / (t_max - t_min) * 2
        print(self.train_data)
        
    def _impute(self, t_mean, is_test=False):
        mat = self.impute_train
        data = self.train_data
        if is_test:
            mat = self.impute_test
            data = self.test_data
        for row in mat:
            for i, idx in zip([0, 2, 3, 4, 5], [0, 1, 2, 3, 4]):
                if len(row[i]) == 0:
                    row[i] = t_mean[idx]
        temp = []
        for row in mat:
            temp.append(self._make_psg(row).get_info())
        impute_mat = np.array([np.array(i) for i in temp])
        data = np.concatenate((data, impute_mat), axis=0)
        if is_test:
            self.test_data = data
        else:
            self.train_data = data
            
    @classmethod
    def select_aj(cls, i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j
    

class Passenger:
    def __init__(self, pclass, sex, age, sibsp, parch):
        self.pclass = pclass
        self.sex = sex
        self.age = age
        self.sibsp = sibsp
        self.parch = parch

    def get_info(self):
        return [self.pclass, self.sex, self.age, self.sibsp, self.parch]

def main():
    titanic = Titanic()
    

if __name__ == "__main__":
    main()
