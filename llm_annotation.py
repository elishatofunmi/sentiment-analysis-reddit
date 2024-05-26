from openai import OpenAI
import requests
import pandas as pd
from decouple import config


GPT_MODEL = "gpt-3.5-turbo-0613"
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + config("OPEN_AI_API_KEY"),
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def inquire_information(message):
#print("message: ", message)
  pchats = [{"role": "user", "content": "hey look at this statement"},
      {"role": "user", "content": "STATEMENT: "+str(message)}, {"role": "system", "content": "Tell me if 'STATEMENT' seems like something written by any of these 3: medical doctor, veterinarian, or others . Don't explain anything, just tell me medical doctor, veterinarian or others"},
            {"role": "system", "content": "you have 3 options: medical doctor, veterinarian and others."}]
  chat_response = chat_completion_request(pchats, None)
  return chat_response.json()["choices"][0]["message"]['content']


RedditTrainData = {"comment": [], "label": []}
RedditTestData = {"comment": [], "label": []}
RedditValData = {"comment": [], "label": []}


def processData(data, logtype='train'):
  try:
    response = inquire_information(str(data[1]))
    if logtype == 'train':
      RedditTrainData['comment'].append(str(data[1]))
      RedditTrainData['label'].append(response.lower())
    elif logtype== 'test': 
      RedditTestData['comment'].append(str(data[1]))
      RedditTestData['label'].append(response.lower())
    else:
      RedditValData['comment'].append(str(data[1]))
      RedditValData['label'].append(response.lower())
  except KeyError as err:
    print("ignore")
    pass
  return


trainlog = [processData(data, logtype= 'train') for i,data in enumerate(tqdm(TrainData))]
# testlog = [processData(data, logtype= 'test') for i,data in enumerate(tqdm(TestData))]
# vallog = [processData(data, logtype= 'val') for i,data in enumerate(tqdm(validationData))]


TrainDF = pd.DataFrame(RedditTrainData)
# TestDF = pd.DataFrame(RedditTestData)
# ValDF = pd.DataFrame(RedditValData)

# TrainDF.to_csv("reddit_train.csv", index=False)
# TestDF.to_csv("reddit_test.csv", index=False)
# ValDF.to_csv("reddit_val.csv", index=False)

label_list = ['medical doctor', 'veterinarian', 'others']
df_train = pd.concat([TrainDF[TrainDF['label'].isin(label_list)]])

# df_train.head()

# np.unique(df_train['label'], return_counts=True)


