"""Util
which is not important to the experiments themselves
"""

import requests


def post_to_email(message):
    """Makes a POST request to a (private) server which parses the request and sends an email with @message as contents

    Used to inform myself when (long running) experiments have exited (successfully or not)
    """
    url = ""  # PINGBACK_URL
    access_key = ""  # PINGBACK_ACCESS_KEY
    if url != "":
        files = {
            "key": (None, access_key),
            "message": (None, message),
        }
        response = requests.post(url, files=files)
        # print(f"POST {url}: {response.status_code}:{response.text}")


# The php script for the server
"""php
<?php
    if (!empty($_POST)) {
        if (!empty($_POST['access-key']) && !empty($_POST['error-message'])){
            if('<PINGBACK_ACCESS_KEY>' == $_POST['access-key']){
                $to      = '<COMMA_SEPARATED_LIST_OF_EMAILS_RECEIVERS>';
                $subject = 'Error';
                $message = $_POST['error-message'];
                $headers = 'From: <YOUR_SERVER_EMAIL_SENDER>'       . "\r\n" .
                            'Reply-To: <YOUR_SERVER_EMAIL_SENDER>' . "\r\n" .
                            'X-Mailer: PHP/' . phpversion();
        
                mail($to, $subject, $message, $headers);
            }
        }
    }
?>
"""
