"""
Email tools module for the email assistant agent.
Contains all IMAP-related tools and utilities.
"""

import logging
import os

from dotenv import load_dotenv
from imap_tools import AND, MailBox
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

# Configure logger for this module
logger = logging.getLogger(__name__)

# IMAP configuration
IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = "INBOX"

# LLM configuration for summarization
CHAT_MODEL = os.getenv("GEMINI_MODEL_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM for summarization
llm = init_chat_model(model=CHAT_MODEL, api_key=GEMINI_API_KEY)


def mail_box_connect():
    """Establish and return an IMAP mailbox connection."""
    try:
        mail_box = MailBox(IMAP_HOST)
        mail_box.login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)
        return mail_box
    except Exception as e:
        logger.error(f"Failed to connect to IMAP server: {str(e)}")
        raise Exception(f"Failed to connect to IMAP server: {str(e)}")


@tool
def list_unread_emails():
    """Return a formatted list of up to 5 unread emails with UID, date, subject, and sender."""
    try:
        with mail_box_connect() as mailbox:
            mailbox.folder.set(IMAP_FOLDER)
            unread_emails = mailbox.fetch(
                limit=5, criteria=AND(seen=False), headers_only=True
            )
            emails = list(unread_emails)
            if not emails:
                return "You have no unread messages"

            # Format emails in a table-like structure
            header = f"{'UID':<8} {'Date':<16} {'Sender':<30} {'Subject':<50}"
            separator = "-" * 104
            rows = [
                f"{mail.uid:<8} "
                f"{mail.date.astimezone().strftime('%Y-%m-%d %H:%M'):<16} "
                f"{(mail.from_ or 'Unknown')[:29]:<30} "
                f"{(mail.subject or 'No Subject')[:49]:<50}"
                for mail in emails
            ]
            response = "\n".join([header, separator] + rows)
            logger.info("list_unread_emails tool executed successfully")
            return response
    except Exception as e:
        error_msg = f"Error fetching emails: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_email_content(uid: str):
    """Get the full content of an email by its UID."""
    try:
        with mail_box_connect() as mailbox:
            mailbox.folder.set(IMAP_FOLDER)
            emails = mailbox.fetch(criteria=f"UID {uid}")
            emails_list = list(emails)

            if not emails_list:
                return f"No email found with UID {uid}"

            email = emails_list[0]
            content = f"From: {email.from_}\n"
            content += f"To: {email.to}\n"
            content += (
                f"Date: {email.date.astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            content += f"Subject: {email.subject}\n"
            content += f"Content:\n{email.text or email.html}\n"

            logger.info(f"get_email_content tool executed for UID {uid}")
            return content
    except Exception as e:
        error_msg = f"Error getting email content for UID {uid}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def mark_email_as_read(uid: str):
    """Mark an email as read by its UID."""
    try:
        with mail_box_connect() as mailbox:
            mailbox.folder.set(IMAP_FOLDER)
            mailbox.flag(uid, [r"\Seen"], True)
            logger.info(f"mark_email_as_read tool executed for UID {uid}")
            return f"Email with UID {uid} marked as read"
    except Exception as e:
        error_msg = f"Error marking email as read for UID {uid}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def summarize_email(uid: str):
    """Get and summarize an email by its UID using AI."""
    try:
        with mail_box_connect() as mailbox:
            mailbox.folder.set(IMAP_FOLDER)
            emails = mailbox.fetch(criteria=f"UID {uid}")
            emails_list = list(emails)

            if not emails_list:
                return f"No email found with UID {uid}"

            email = emails_list[0]

            # Prepare email content for summarization
            email_content = f"""
            From: {email.from_}
            To: {email.to}
            Date: {email.date.astimezone().strftime("%Y-%m-%d %H:%M:%S")}
            Subject: {email.subject}
            
            Content:
            {email.text or email.html or "No content available"}
            """

            # Create a summarization prompt
            summary_prompt = f"""
            Please provide a concise summary of this email.             
            {email_content}
            """

            # Use the LLM to generate summary
            summary_response = llm.invoke([{"role": "user", "content": summary_prompt}])

            result = f"Summary of Email UID {uid}:\n"
            result += f"Subject: {email.subject}\n"
            result += f"Summary:\n{summary_response.content}"

            logger.info(f"summarize_email tool executed for UID {uid}")
            return result

    except Exception as e:
        error_msg = f"Error summarizing email for UID {uid}: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Export all tools
EMAIL_TOOLS = [
    list_unread_emails,
    get_email_content,
    mark_email_as_read,
    summarize_email,
]
