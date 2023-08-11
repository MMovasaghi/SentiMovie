import sys
import asyncio
sys.path.append("..")
import model.model_back as model_back
myClassifier= model_back.MyClassifier()

def log(id, title, is_passed, discription=None):
    print(f"Test[{id}][{title}]: {is_passed} ({discription})")

async def testing():
    id = 0
    is_failed = False
    # ====================================================================================================
    result = await myClassifier.single_prediction(text = "I hate this movie")
    is_passed = "Failed"
    if result['sentiment'] == 'Negative':
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment is Negative", is_passed=is_passed, discription="result['sentiment'] == 'Negative'")

    is_passed = "Failed"
    if result['probability'] >= 0.7:
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment probability", is_passed=is_passed, discription="result['probability'] >= 0.7")

    # ====================================================================================================
    result = await myClassifier.single_prediction(text = "I like this movie")
    is_passed = "Failed"
    if result['sentiment'] == 'Positive':
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment is Positive", is_passed=is_passed, discription="result['sentiment'] == 'Positive'")

    is_passed = "Failed"
    if result['probability'] >= 0.7:
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment probability", is_passed=is_passed, discription="result['probability'] >= 0.7")
    # ====================================================================================================

    # ====================================================================================================
    results = await myClassifier.batch_prediction(texts = 
                                                  [
                                                      "I hate this movie",
                                                      "I hate this movie",
                                                      "I hate this movie",
                                                      "I hate this movie",
                                                      "I hate this movie",
                                                      "I like this movie",
                                                      "I like this movie",
                                                      "I like this movie",
                                                      "I like this movie",
                                                      "I like this movie",
                                                  ])
    is_passed = "Failed"
    for idx, result in enumerate(results['detail']):
        target = "Negative"
        if idx >= 5:
            target = "Positive"
        if result['sentiment'] == target:
            is_passed = "Passed"
        else:
            is_passed = "Failed"
            is_failed = True
            break
    id += 1
    log(id=id, title="Sentiment on Batch", is_passed=is_passed, discription="5 negative and 5 positive")
    is_passed = "Failed"
    for result in results['detail']:
        if result['probability'] >= 0.7:
            is_passed = "Passed"
        else:
            is_passed = "Failed"
            is_failed = True
            break
    id += 1
    log(id=id, title="Probability on Batch", is_passed=is_passed, discription="5 negative and 5 positive with > 0.7 prob")
    # ====================================================================================================

    # ====================================================================================================
    result = await myClassifier.explainable_prediction(text = "I like this movie")
    is_passed = "Failed"
    if result['sentiment'] == 'Positive':
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment is Positive [explainable]", is_passed=is_passed, discription="result['sentiment'] == 'Positive'")

    is_passed = "Failed"
    if result['probability'] >= 0.7:
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment probability [explainable]", is_passed=is_passed, discription="result['probability'] >= 0.7")
    # ====================================================================================================

    # ====================================================================================================
    result = await myClassifier.explainable_prediction(text = "I hate this movie")
    is_passed = "Failed"
    if result['sentiment'] == 'Negative':
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment is Negative [explainable]", is_passed=is_passed, discription="result['sentiment'] == 'Negative'")

    is_passed = "Failed"
    if result['probability'] >= 0.7:
        is_passed = "Passed"
    else:
        is_passed = "Failed"
        is_failed = True
    id += 1
    log(id=id, title="Sentiment probability [explainable]", is_passed=is_passed, discription="result['probability'] >= 0.7")
    # ====================================================================================================

    if is_failed:
        raise Exception("At least one test was failed.")



if __name__ == '__main__':
    print("Testing model")
    asyncio.run(testing())