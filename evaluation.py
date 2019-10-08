class ConsecutiveSequencesDetectionEvaluation(object):
    def __init__(self, pos_class, num_sequence=1):
        self.pos_class = pos_class
        self.num_sequence = num_sequence

    def eval(self, prediction):
        count = 0
        for label in prediction:
            if label == self.pos_class:
                count += 1
            else:
                count = 0
            if count >= self.num_sequence:
                return True
        return False






