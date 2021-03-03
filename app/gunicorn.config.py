_log_file = '../gunicorn.logs.log'
worker_class = "gevent"
reload = True
bind = "127.0.0.1:5000"
loglevel = "debug"
errorlog = _log_file
accesslog = _log_file
