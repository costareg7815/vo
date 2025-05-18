<?php
if(isset($_post['bs'])){
    $host="localhost";
    $password=$_post['uip'];
    $user_name=$_post['up'];
    $db_name="";

}
$conn= mysqli_connect($host,"root","",$db_name);




?>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign in</title>
</head>
<body>
    <p>sign</p>
    <div id=uin>
        <label for=up>put your name</label>
        <input name=up type="text" placeholder="text here your name" required>
    </div>
    <div id="pui">
        <label for="uip">put your password</label>
        <input type="password" name="uip"placeholder="text here your password" >
    </div>
    <div for="bs">
    <button name="bs" value="send" >send </button>
    </div>  
</body></html>




































