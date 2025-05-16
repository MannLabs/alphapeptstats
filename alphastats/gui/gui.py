def run():
    import os
    import sys

    from streamlit.web import cli as stcli

    file_path = os.path.join(os.path.dirname(__file__), "AlphaPeptStats.py")

    args = [
        "streamlit",
        "run",
        file_path,
        "--global.developmentMode=false",
    ]

    # # this is to avoid 'AxiosError: Request failed with status code 403' locally, cf. https://github.com/streamlit/streamlit/issues/8983
    # # Do not use this in production!
    if os.environ.get("DISABLE_XSRF", 0):
        args.extend(["--server.enableXsrfProtection", "false"])

    sys.argv = args

    sys.exit(stcli.main())
