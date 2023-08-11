import transformers
import shap


class MyClassifier():
    def __init__(self):
        self.round_number = 4
        self.classifier = transformers.pipeline(model='Movasaghi/finetuning-sentiment-rottentomatoes', 
                                    return_all_scores=True)
        self.explainer = shap.Explainer(self.classifier)


    async def batch_prediction(self, texts):
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