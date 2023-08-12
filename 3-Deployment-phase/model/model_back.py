import transformers
import shap


class MyClassifier():
    def __init__(self):
        """
        Initializes an instance of the MyClassifier class.

        Attributes:
        - round_number: The number of decimal places to round the prediction probabilities.
        - classifier: A sentiment analysis pipeline using a pre-trained model from Hugging Face Transformers.
        - explainer: A SHAP explainer object used for generating model explanations.
        """
        self.round_number = 4
        self.classifier = transformers.pipeline(model='Movasaghi/finetuning-sentiment-rottentomatoes', 
                                    return_all_scores=True)
        self.explainer = shap.Explainer(self.classifier)


    async def batch_prediction(self, texts):
        """
        Perform batch sentiment analysis prediction on a list of texts.

        Args:
        - texts (list): A list of text inputs for sentiment analysis.

        Returns:
        - dict: A dictionary containing sentiment analysis results.
        - "sentiment": The overall sentiment prediction for the batch ("Positive" or "Negative").
        - "score": The proportion of positive sentiment predictions in the batch (rounded).
        - "detail" (list): A list of dictionaries containing detailed results for each text input.
            - "text": The input text.
            - "sentiment": The predicted sentiment for the text ("Positive" or "Negative").
            - "probability": The predicted sentiment probability (rounded).
        """
        output= self.classifier(texts)
        results = {
            "sentiment": None,
            "score": None,
            "detail": []
            }
        positive_count = 0
        negative_count = 0
        for i in range(len(output)):
            result = {
                "text": texts[i], 
                "sentiment": None,
                "probability": None
                }
            if output[i][0]['score'] > output[i][1]['score']:
                result['sentiment'] = "Negative"
                result['probability'] = round(output[i][0]['score'], self.round_number)
                negative_count += 1
            else:
                result['sentiment'] = "Positive"
                result['probability'] = round(output[i][1]['score'], self.round_number)
                positive_count += 1
            results["detail"].append(result)
        results['sentiment'] = "Positive" if positive_count > negative_count else "Negative"
        results['score'] = round(positive_count / (negative_count + positive_count), self.round_number)
        return results



    async def single_prediction(self, text):
        """
        Perform sentiment analysis prediction on a single text.

        Args:
        - text (str): The text input for sentiment analysis.

        Returns:
        - dict: A dictionary containing sentiment analysis result for the input text.
        - "text": The input text.
        - "sentiment": The predicted sentiment ("Positive" or "Negative").
        - "probability": The predicted sentiment probability (rounded).
        """
        output= self.classifier(text)
        result = {
            "text": text, 
            "sentiment": None,
            "probability": None
            }
        if output[0][0]['score'] > output[0][1]['score']:
            result['sentiment'] = "Negative"
            result['probability'] = round(output[0][0]['score'], self.round_number)
        else:
            result['sentiment'] = "Positive"
            result['probability'] = round(output[0][1]['score'], self.round_number)
        return result


    async def explainable_prediction(self, text):
        """
        Perform explainable sentiment analysis prediction on a single text.

        Args:
        - text (str): The text input for sentiment analysis.

        Returns:
        - dict: A dictionary containing sentiment analysis result and SHAP explanation for the input text.
        - "text": The input text.
        - "sentiment": The predicted sentiment ("Positive" or "Negative").
        - "probability": The predicted sentiment probability (rounded).
        - "diagram": A diagram in HTML format representing the SHAP explanation for the prediction.
        """
        output = self.classifier(text)
        shap_values = self.explainer([text])
        exp_html = shap.plots.text(shap_values[:,:,"LABEL_1"], display=False)
        result = {
            "text": text, 
            "sentiment": None,
            "probability": None,
            "diagram": exp_html
            }
        if output[0][0]['score'] > output[0][1]['score']:
            result['sentiment'] = "Negative"
            result['probability'] = round(output[0][0]['score'], self.round_number)
        else:
            result['sentiment'] = "Positive"
            result['probability'] = round(output[0][1]['score'], self.round_number)
        return result