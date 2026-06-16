from utils.angles import calculate_angle

class BicepAnalyzer:

    def analyze(self,lm):

        feedback=[]

        elbow_angle=calculate_angle(
            lm["LEFT_SHOULDER"],
            lm["LEFT_ELBOW"],
            lm["LEFT_WRIST"]
        )

        if elbow_angle>60:
            feedback.append(
                "Complete the curl"
            )

        return feedback