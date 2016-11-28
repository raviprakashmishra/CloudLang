function submitTextForNLP() {
    var inputText = $("#input-text").val();
    console.log(inputText);
    var request = $.ajax({
        method: "GET",
        url: "http://localhost:8080/CloudLangService/rest/parse/" + inputText
    });

    request.done(function(data){
        console.log("data", data);
    });
}