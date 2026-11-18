import sys

import logging

def handle_exception(error, error_description: sys):
    """
    Handle exceptions by printing the error message and exiting the program.

    Args:
        error (Exception): The exception that occurred.
        error_description (str): A description of the error context.
    """
    _,_,exc_tb = sys.exc_info()
    filename= exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno    
    error_mesage = f"Error occurred in file: {filename}, line: {lineno}, error:{error}, description: {error_description}"
    
    return error_mesage

class CustomException(Exception):
    """
    Custom exception class that extends the built-in Exception class.

    Args:
        error_message (str): The error message to be associated with the exception.
        error_details (str): A description of the error context.
    """
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = handle_exception(error_message, error_details)
    
    def __str__(self):
        return self.error_message
    
    

if __name__ == "__main__":
    try:
       a = 1 / 0  
    except Exception as e:        
        logging.info("CustomException occurred: %s", e)
        raise CustomException(e,sys)