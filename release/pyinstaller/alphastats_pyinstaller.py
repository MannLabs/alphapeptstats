if __name__ == "__main__":
    try:
        import alphastats
        import multiprocessing
        multiprocessing.freeze_support()
      
    except ModuleNotFoundError:
        pass
    from alphastats.gui import run
    run()


