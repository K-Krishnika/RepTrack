from utils.angles import calculate_angle

class PushupAnalyzer:

    def analyze(self,lm):

        feedback=[]

        elbow_angle=calculate_angle(
            lm["LEFT_SHOULDER"],
            lm["LEFT_ELBOW"],
            lm["LEFT_WRIST"]
        )

        if elbow_angle>100:
            feedback.append(
                "Lower your chest further"
            )

        return feedback