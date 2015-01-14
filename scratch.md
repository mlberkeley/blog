---
layout: page
title: Scratch
---

Testing new features, behaviors, and visuals.

<script>
  var ShortMonthForIndex = { 0: "Jan", 1: "Feb", 2: "Mar", 3: "Apr", 4: "May", 5: "Jun", 6: "Jul", 7: "Aug", 8: "Sep", 9: "Oct", 10: "Nov", 11: "Dec" };
  var IssueUrl = "";
  var CommentsArray = [];

  function formatComment(userAvatarUrl, userHtmlUrl, userLogin, commentBodyHtml, commentTimeStamp) {
    var commentDate = new Date(commentTimeStamp);
    var shortMonth = ShortMonthForIndex[commentDate.getMonth()];
    var commentHtml = '<img src="' + userAvatarUrl + '" height="42" />';
    commentHtml += '<a href="' + userHtmlUrl + '">' + userLogin + '</a> <small>commented on ' + commentDate.getDate() + ' ' + shortMonth + ' ' + commentDate.getFullYear() + '</small>';
    commentHtml += commentBodyHtml;
    commentHtml += '<hr />';
    return commentHtml;
  }

  function presentAllComments(allComments) {
    var allCommentsHtml = allComments.length === 0 ? "<p>No comments</p>" : "";
    for (var i = 0; i < allComments.length; i++) {
      var user = allComments[i].user;
      allCommentsHtml += formatComment(user.avatar_url, user.html_url, user.login, allComments[i].body_html, allComments[i].updated_at);
    }

    document.getElementById("all_comments").innerHTML = allCommentsHtml;
    document.getElementById("load_comments_button").innerHTML = "";

    var leaveCommentUrl = allComments.length > 0 ? allComments.pop().html_url : IssueUrl;
    document.getElementById("add_comment_link").innerHTML = '<a href="' + leaveCommentUrl + '">Leave a comment</a>';
  }

  function getGitHubApiRequestWithCompletion(url, completion)
  {
    var gitHubRequest = new XMLHttpRequest(); 
    gitHubRequest.open("GET", url, true);
    gitHubRequest.setRequestHeader("Accept", "application/vnd.github.v3.html+json");
    gitHubRequest.onreadystatechange = function() {
      if (gitHubRequest.readyState != 4 || gitHubRequest.status != 200) return;
      completion(gitHubRequest);
    };
    gitHubRequest.send();
  }

  function onCommentsUpdated(commentRequest)
  {
    CommentsArray = CommentsArray.concat(JSON.parse(commentRequest.responseText));
    var commentsPages = commentRequest.getResponseHeader("Link");
    if (commentsPages) {
      var commentsLinks = commentsPages.split(",");
      for (var i = 0; i < commentsLinks.length; i++) {
        if (commentsLinks[i].search('rel="next"') > 0) {
          var linkStart = commentsLinks[i].search("<");
          var linkStop = commentsLinks[i].search(">");
          var nextLink = commentsLinks[i].substring(linkStart + 1, linkStop);
          getGitHubApiRequestWithCompletion(nextLink, onCommentsUpdated);
          return;
        }
      }
      presentAllComments(CommentsArray);
    }
    else {
      presentAllComments(CommentsArray);
    }
  }

  function onSearchComplete(searchRequest) {
    var searchResults = JSON.parse(searchRequest.responseText);
    if (searchResults.total_count === 1) {
      IssueUrl = searchResults.items[0].html_url;
      getGitHubApiRequestWithCompletion(searchResults.items[0].comments_url, onCommentsUpdated);
    }
    else {
      presentAllComments(CommentsArray);
    }
  }

  function findAndPresentComments(userName, repositoryName, issueTitle) {
    var safeQuery = encodeURI(issueTitle);
    var seachQueryUrl = "https://api.github.com/search/issues?q=" + safeQuery + "+repo:" + userName + "/" + repositoryName + "+type:issue+in:title";
    getGitHubApiRequestWithCompletion(seachQueryUrl, onSearchComplete)
  }
</script>

<div id="all_comments"></div>

<div id="load_comments_button">
  <button class="search-submit" onclick='findAndPresentComments("joyent", "node", "net.js - possible EventEmitter memory leak detected")'>Show Comments</button> 
</div>

<div id="add_comment_link"></div>
