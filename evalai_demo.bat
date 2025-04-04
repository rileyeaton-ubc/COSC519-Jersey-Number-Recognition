set /p eval_ai_token=<aidan_token.txt

@echo off
evalai set_token %eval_ai_token%
@echo on

evalai submission 503140 result

TIMEOUT /T 10

evalai submission 502408 result