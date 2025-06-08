from datetime import datetime

import requests
from openai import APITimeoutError, AuthenticationError, BadRequestError, ConflictError, \
    InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError, UnprocessableEntityError
import time

DAILY_RATE_LIMIT = 75000
HOURLY_RATE_LIMIT = 3000
MINUTE_RATE_LIMIT = 60
SECOND_RATE_LIMIT = 2


def handle_api_errors(call_times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            manual_retries = 0
            unexpected_rate_limit_retries = 0
            connection_retries = 0
            while True:  # Loop indefinitely until the function succeeds
                try:
                    now = time.time()
                    call_times.append(now)
                    return func(*args, **kwargs)
                except AuthenticationError as e:
                    print(
                        f"\nYour API key or token was invalid, expired, or revoked. Check your API key or token. Error: {e}")
                    return e
                except BadRequestError as e:
                    print(
                        f"\nYour request was malformed or missing some required parameters. Check the specific API method documentation you are calling. Error: {e}")
                    return e
                except ConflictError as e:
                    print(
                        f"\nThe resource was updated by another request. Try to update the resource again. Error: {e}")
                    return e
                except NotFoundError as e:
                    print(
                        f"\nRequested resource does not exist. Ensure you enter the correct resource identifier. Error: {e}")
                    return e
                except PermissionDeniedError as e:
                    print(
                        f"\nYou don't have access to the requested resource. Ensure you are using the correct API key, organization ID, and resource ID. Error: {e}")
                    return e

                except RateLimitError as e:
                    now = time.time()

                    # Determine time boundaries
                    one_hour_ago = now - 3600
                    start_of_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

                    # Filter timestamps to:
                    # 1. Keep timestamps younger than one hour (regardless of date)
                    # 2. Keep timestamps older than one hour if they are from today
                    relevant_call_times = [
                        ts for ts in call_times
                        if ts >= one_hour_ago or (start_of_today <= ts < now)
                    ]

                    if len(relevant_call_times) > DAILY_RATE_LIMIT:
                        # sleep until 12am EST
                        time_to_sleep = 86400 - (now % 86400)
                        print(
                            f"\nDaily rate limit exceeded. Waiting for {time_to_sleep} seconds before retrying. Error: {e}")
                        time.sleep(time_to_sleep)
                    elif len(relevant_call_times) > HOURLY_RATE_LIMIT and now - relevant_call_times[-HOURLY_RATE_LIMIT] < 3600:
                        time_to_sleep = relevant_call_times[-HOURLY_RATE_LIMIT] + 3600 - now
                        print(
                            f"\nHourly rate limit exceeded. Waiting for {time_to_sleep} seconds before retrying. Error: {e}")
                        time.sleep(time_to_sleep)
                    elif len(relevant_call_times) > MINUTE_RATE_LIMIT and now - relevant_call_times[-MINUTE_RATE_LIMIT] < 60:
                        time_to_sleep = relevant_call_times[-MINUTE_RATE_LIMIT] + 60 - now
                        print(
                            f"\nMinute rate limit exceeded. Waiting for {time_to_sleep} seconds before retrying. Error: {e}")
                        time.sleep(time_to_sleep)
                    elif len(relevant_call_times) > SECOND_RATE_LIMIT and now - relevant_call_times[-SECOND_RATE_LIMIT] < 1:
                        time_to_sleep = 1
                        print(
                            f"\nSecond rate limit exceeded. Waiting for 1 second before retrying. Error: {e}")
                        time.sleep(time_to_sleep)
                    else:
                        unexpected_rate_limit_retries += 1
                        if unexpected_rate_limit_retries < 10:
                            time_to_sleep = 2**unexpected_rate_limit_retries
                            print(f"\nUnexpected Rate Limit Error: {e}. Waiting for {time_to_sleep} seconds before retrying.")
                            time.sleep(time_to_sleep)
                        else:
                            print(f"\nExceeded the maximum number of retries for unexpected rate limit errors. Exiting. Error: {e}")
                            return e


                except (APITimeoutError, InternalServerError, UnprocessableEntityError) as e:
                    if manual_retries < 2:
                        manual_retries += 1
                        if isinstance(e, APITimeoutError):
                            print(f"\nRequest timed out. Retrying after 10 seconds. Error: {e}")
                        else:
                            print(f"\nIssue on OpenAI's side. Retrying after 10 seconds. Error: {e}")
                        time.sleep(10)
                    else:
                        print(f"\nExceeded the maximum number of retries. Exiting. Error: {e}")
                        return e
                except requests.exceptions.ConnectionError as e:
                    if connection_retries < 3:
                        connection_retries += 1
                        print(f"\nConnection error occurred. Retrying in 10 minutes... Attempt {connection_retries}/3. Error: {e}")
                        time.sleep(600)
                    else:
                        print(f"\nExceeded the maximum number of retries for connection errors. Exiting. Error: {e}")
                        return e
                except Exception as e:
                    print(f"\nAn unexpected error occurred. Error: {e}")
                    return e

        return wrapper

    return decorator
