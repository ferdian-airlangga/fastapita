import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def kirim_email(email_recipient,nama_kandidat,posisi_dilamar) :

    email_sender = "cvscreeningsystem@cvscreeningsystem.awsapps.com"
    message = MIMEMultipart("alternative")
    message["Subject"] = "Pengumuman Hasil CV Screening"
    message["From"] = email_sender
    message["To"] = email_recipient

    html = f"""
    <html>
      <head>
        <style>
          body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
          }}
          .container {{
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
          }}
          h1 {{
            font-size: 24px;
            color: #333;
          }}
          p {{
            margin-bottom: 10px;
          }}
          .highlight {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
          }}
          .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            text-align: center;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <p>Halo {nama_kandidat},</p>
          <p>Kami ingin memberitahu Anda tentang hasil seleksi screening CV yang telah dilakukan. Setelah mempertimbangkan dengan seksama, kami ingin memberitahukan bahwa:</p>
          <div class="highlight">
            <p><strong>Nama Kandidat:</strong> {nama_kandidat}</p>
            <p><strong>Posisi yang Dilamar:</strong> {posisi_dilamar}</p>
            <p><strong>Status:</strong> Lanjut ke Tahap Berikutnya</p>
          </div>
          <p>Kami mengucapkan selamat kepada {nama_kandidat}! Kami akan menghubungi Anda segera untuk memberikan informasi lebih lanjut mengenai tahap selanjutnya dalam proses seleksi.</p>
          <p>Jika Anda memiliki pertanyaan atau butuh informasi tambahan, jangan ragu untuk menghubungi kami.</p>
          <div class="footer">
            <p>Terima kasih,</p>
            <p>Tim Rekrutmen</p>
          </div>
        </div>
      </body>
    </html>
    """

    part2 = MIMEText(html, "html")
    message.attach(part2)
    
    server = smtplib.SMTP_SSL('smtp.mail.us-east-1.awsapps.com', 465)
    server.ehlo()
    server.login("cvscreeningsystem@cvscreeningsystem.awsapps.com","@CVScreeningsystem123")
    text = message.as_string()
    server.sendmail(email_sender, email_recipient, text)
    server.quit()
 


#email_recipient = "ferdianairlangga11@gmail.com"
#nama_kandidat = "toni hartono"
#posisi_dilamar = "software engineer"
#kirim_email(email_recipient,nama_kandidat,posisi_dilamar)