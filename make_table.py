from sqlalchemy import create_engine
from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import mysql.connector


# DB 연결 정보
db = {"user": "root", "password": "1234", "host": "localhost", "port": 3306, "database": "cmg"}

# DB URL 생성
DB_URL = (
    f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
)

# SQLAlchemy 엔진 생성
engine = create_engine(DB_URL, pool_pre_ping=True)

# DB 연결 테스트
try:
    with engine.connect() as connection:
        print("MySQL 서버 연결 성공!")
except Exception as e:
    print(f"MySQL 서버 연결 실패: {str(e)}")


Base = declarative_base()

# User 모델 정의


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)


# # CM 모델
# class customized_model(Base):
#     __tablename__ = 'customized_model'

#     cid = Column(Integer, primary_key=True,autoincrement=True)
#     user_id = Column(String(255))
#     uuId = Column(String(255))

# # processing
# class cm_processing(Base):
#     __tablename__ = 'cm_processing'

#     pid = Column(Integer, primary_key=True,autoincrement=True)
#     uuId = Column(String(255))
#     status = Column(String(255))


class CustomizedModel(Base):
    __tablename__ = 'Customized_Model'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)

    def __repr__(self):
        return f'<CustomizedModel {self.uuid}>'

class CMProcessing(Base):
    __tablename__ = 'CMProcessing'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    status = Column(String, nullable=False)

    def __repr__(self):
        return f"<CMProcessing(id={self.id}, uuid='{self.uuid}', status='{self.status}')>"

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)


def connect_DB():
    try:
        session = Session()

        users = session.query(User).all()

        print("Users:")
        for user in users:
            print(f"ID: {user.id},  Email: {user.email}")

        print("\n")

        # CM 모델 테이블 조회
        cm_models = session.query(customized_model).all()
        print("Customized Models:")
        for model in cm_models:
            print(f"CID: {model.cid}, User ID: {model.uid}, UUID: {model.independentKey}")

        print("\n")

        # Session 모델 테이블 조회
        cm_sessions = session.query(cm_processing).all()
        print("CM Processing Sessions:")
        for session in cm_sessions:
            print(f"PID: {session.pid}, UUID: {session.independentKey}, Status: {session.status}")

    except Exception as e:
        print("실패", e)


#connect_DB()