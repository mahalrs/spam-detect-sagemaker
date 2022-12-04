import json
import os

from email import message_from_string
from email.parser import Parser
from email.policy import default

from botocore.exceptions import ClientError
import boto3

from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

REGION = os.environ['REGION']
PREDICTION_ENDPOINT = os.environ['PREDICTION_ENDPOINT']


def lambda_handler(event, context):
    print('Received event: ' + json.dumps(event))

    email = get_email(event)
    print(email)

    msg = format_email_msg(email['msg'])

    # Call prediction endpoint
    label, prob = predict(msg)
    prob *= 100

    res = prepare_response(email['received'], email['subject'], email['msg'],
                           label, prob)
    print(res)

    # send email
    sub = 'Reply: ' + email['subject']
    send_email(sub, res, email['from'], email['to'])

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Inference Lambda!')
    }


def predict(msg):
    messages = [msg]
    vocabulary_length = 9013
    one_hot_test_messages = one_hot_encode(messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages,
                                                vocabulary_length)
    payload = json.dumps(encoded_test_messages.tolist())

    print('Invoking Prediction Endpoint')

    client = boto3.client('sagemaker-runtime')
    res = client.invoke_endpoint(EndpointName=PREDICTION_ENDPOINT,
                                 Body=payload,
                                 ContentType='application/json')
    print(res)

    result = json.loads(res['Body'].read().decode())
    print(result)

    return result['predicted_label'][0][0], result['predicted_probability'][0][
        0]


def get_email(event):
    object_key = event['Records'][0]['s3']['object']['key']
    bucket = event['Records'][0]['s3']['bucket']['name']

    s3client = boto3.client('s3')
    obj = s3client.get_object(Bucket=bucket, Key=object_key)
    raw_msg = obj['Body'].read().decode('utf-8')

    headers = Parser(policy=default).parsestr(raw_msg)
    to_email = headers['to']
    from_email = headers['from']
    date = headers['date']
    subject = headers['subject']

    messages = []
    for payload in message_from_string(raw_msg).get_payload():
        messages.append(payload.get_payload())

    return {
        'to': to_email,
        'from': from_email,
        'subject': subject,
        'received': date,
        'msg': messages[0],
    }


def format_email_msg(msg):
    return msg.replace('\r', '').replace('\n', ' ')


def prepare_response(received, subject, msg, label, prob):
    res = 'We received your email sent at ' + received
    res += ' with the subject "' + subject + '".\r\n\r\n'

    res += 'Here is a 240 character sample of the email body:\r\n' + msg[:240] + '\r\n\r\n'

    res += 'The email was categorized as ' + str(label)
    res += ' with a ' + str(prob) + '% confidence.'

    return res


def send_email(sub, body, to, sender):
    charset = 'UTF-8'

    sesclient = boto3.client('ses', region_name=REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = sesclient.send_email(
            Destination={
                'ToAddresses': [to,],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': charset,
                        'Data': body,
                    },
                },
                'Subject': {
                    'Charset': charset,
                    'Data': sub,
                },
            },
            Source=sender,
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print('Email sent! Message ID:')
        print(response['MessageId'])
