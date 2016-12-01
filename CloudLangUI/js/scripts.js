function submitTextForNLP() {
    $("#parseRow").hide(300);
    $("#entityRow").hide(300);
    $("#sentimentRow").hide(300);
    $("#loadingRow").show(400);
    var inputText = $("#input-text").val();
    console.log(inputText);
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log("parse response", JSON.parse(this.responseText));
            var depTree = drawTree(document.getElementById("svgElement"), JSON.parse(this.responseText).parsedResult);
            console.log("depTree", depTree);
            if($("#entityRow").is(":visible") /*&& $("#sentimentRow").is(":visible")*/) {
                $("#loadingRow").hide(300);
            }
            $("#parseRow").show(400);
        }
    };
    xhttp.open("GET", "http://localhost:8080/CloudLangService/rest/parse/" + inputText, true);
    xhttp.send();

    var xhttp2 = new XMLHttpRequest();
    xhttp2.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log("ner response", JSON.parse(this.responseText));
            var response = JSON.parse(this.responseText).nerResult;
            var words = response.split(" ");
            var resultString = '';
            for(var i = 0; i < words.length; i++) {
                if(words[i].includes("__")) {
                    var wordsNtags = words[i].split("__");
                    if(wordsNtags[1].includes("PER")) {
                        resultString += ' <mark data-entity="person">' + wordsNtags[0] + '</mark>';
                    } else if(wordsNtags[1].includes("LOC")) {
                        resultString += ' <mark data-entity="loc">' + wordsNtags[0] + '</mark>';
                    } else if(wordsNtags[1].includes("ORG")) {
                        resultString += ' <mark data-entity="org">' + wordsNtags[0] + '</mark>';
                    } else {
                        resultString += ' ' + wordsNtags[0];
                    }
                } else {
                    resultString += ' ' + words[i];
                }
            }
            console.log("resultString", resultString);
            document.getElementById("entityElement").innerHTML = resultString;
            if($("#parseRow").is(":visible") /*&& $("#sentimentRow").is(":visible")*/) {
                $("#loadingRow").hide(300);
            }
            $("#entityRow").show(400);
        }
    };
    xhttp2.open("GET", "http://localhost:8080/CloudLangService/rest/entities/" + inputText, true);
    xhttp2.send();
}