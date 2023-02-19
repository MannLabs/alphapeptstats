if __name__ == "__main__":
    
    import alphastats
    import multiprocessing
    multiprocessing.freeze_support()
   # alphastats.gui.gui.run()
   # alphastats.cli.run()
    print(alphastats.gui.__dict__.keys())
    ##from alphastats import cli
    #cli.run()
    from alphastats.gui import run
    run()
   # alphastats.gui.gui.run()