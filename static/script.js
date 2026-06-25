setInterval(async ()=>{

    let response =
    await fetch("/predict");

    let data =
    await response.json();

    document.getElementById(
        "label"
    ).innerHTML =
    data.label;

    document.getElementById(
        "confidence"
    ).innerHTML =
    data.confidence + "%";

},500);