# samokat
### клонировать репозиторий
 git clone https://github.com/amsurex/samokat.git
 cd samokat
### cобрать образ, запустить контейнер
 docker build -t my-flask-app .
 docker run -p 5000:5000 my-flask-app

### пример как получить предсказание
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "It has a slight fishy smell, but my dog loves it added to her dry food along with warm water.She wont eat her food in the morning (breakfast) unless this product is added and soaked in water. Its a great price and I think it has actually helped keep her skin and coat healthy (along with her Canidae)."}' 
### выдает 
{"Cat1":"pet supplies","Cat2":"dogs","Cat3":"health supplies"}
