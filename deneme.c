// C
sunucu
kodu

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <sql.h>
# include <sqlext.h>

# define PORT 8080

// SQL
bağlantısı
için
gerekli
değişkenler
SQLHANDLE
sqlenvhandle;
SQLHANDLE
sqlconnectionhandle;
SQLHANDLE
sqlstatementhandle;
SQLRETURN
retcode;

// SQL
sorgusu
için
değişkenler
SQLCHAR * sqlquery = (SQLCHAR *)
"SELECT * FROM Kullanicilar WHERE KullaniciAdi = ? AND Sifre = ?";
SQLCHAR
kullanicilar[256];

int
main()
{
    int
server_fd, new_socket, valread;
struct
sockaddr_in
address;
int
opt = 1;
int
addrlen = sizeof(address);
char
buffer[1024] = {0};
char * hello = "Hello from server";

// SQL
bağlantısı
oluşturma
retcode = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, & sqlenvhandle);
retcode = SQLSetEnvAttr(sqlenvhandle, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)
SQL_OV_ODBC3, 0);
retcode = SQLAllocHandle(SQL_HANDLE_DBC, sqlenvhandle, & sqlconnectionhandle);
retcode = SQLDriverConnect(sqlconnectionhandle, NULL, (SQLCHAR *)
"DSN=my_sql_server_dsn;UID=my_username;PWD=my_password;", SQL_NTS, NULL, 0, NULL, SQL_DRIVER_COMPLETE);

// Sunucu
soketi
oluşturma
if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
exit(EXIT_FAILURE);
}

// Soket
seçeneklerini
ayarlama
if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, & opt, sizeof(opt))) {
    perror("setsockopt");
exit(EXIT_FAILURE);
}

address.sin_family = AF_INET;
address.sin_addr.s_addr = INADDR_ANY;
address.sin_port = htons(PORT);

// Soketi
bağlama
if (bind(server_fd, (struct sockaddr *) & address, sizeof(address)) < 0)
{
    perror("bind failed");
exit(EXIT_FAILURE);
}

// Dinleme
başlatma
if (listen(server_fd, 3) < 0)
{
    perror("listen");
exit(EXIT_FAILURE);
}

if ((new_socket = accept(server_fd, (struct sockaddr *) & address, (socklen_t *) & addrlen)) < 0) {
perror("accept");
exit(EXIT_FAILURE);
}

// İstemciden gelen veriyi okuma
valread = read( new_socket, buffer, 1024);
printf("%s\n", buffer );
send(new_socket, hello, strlen(hello), 0 );
printf("Hello message sent\n");

// SQL bağlantısını kapatma
SQLFreeHandle(SQL_HANDLE_STMT, sqlstatementhandle);
SQLDisconnect(sqlconnectionhandle);
SQLFreeHandle(SQL_HANDLE_DBC, sqlconnectionhandle);
SQLFreeHandle(SQL_HANDLE_ENV, sqlenvhandle);

return 0;
}