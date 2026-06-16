from utils.angles import calculate_angle

class SquatAnalyzer:

    def analyze(self,lm):

        feedback=[]

        left_hip=lm["LEFT_HIP"]
        left_knee=lm["LEFT_KNEE"]
        left_ankle=lm["LEFT_ANKLE"]

        knee_angle=calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        if knee_angle>100:
            feedback.append("Go lower")

        return feedback