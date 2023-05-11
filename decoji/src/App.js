import './App.css';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Card from 'react-bootstrap/Card';
//import SentimentBar from './Bar';
import Stack from 'react-bootstrap/Stack';
import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';

function App() {
  const [text, updateText] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const dummyQuery = (text, axis, setFn) => setFn(Math.round(100 * Math.random()))

  async function fetchPred(text, axis, setFn) {
    //fetch(`http://127.0.0.1:5000/${axis}/${text}`)
    fetch(`http://127.0.0.1:5000/${axis}/${text}`)
      .then(response => response.json())
      .then(resp => {
        console.log(resp)
        setFn(100*resp[0])
        setLoading(false)
        return resp
      })
      .catch(console.error);
  }

  const sentiments = [
    { "axis": "positivity", "labels": ["negative", "positive"], "color": "success", "val": React.useState(50) },
    { "axis": "sincerity", "labels": ["sarcastic", "sincere"], "color": "info", "val": React.useState(50) },
    { "axis": "concreteness", "labels": ["ambiguous", "concrete"], "color": "warning", "val": React.useState(50) },
    { "axis": "intensity", "labels": ["calm", "intense"], "color": "danger", "val": React.useState(50) }
  ]

  function TextInput() {
    const onFormSubmit = e => {
      e.preventDefault()
      const formData = new FormData(e.target),
            formDataObj = Object.fromEntries(formData.entries())
      console.log(formDataObj)
      handleSubmit(formDataObj.text, fetchPred)
    }

    return (
    <Form onSubmit={onFormSubmit}>
      <Form.Group className="mb-3">
        <Form.Control placeholder="Enter text here" defaultValue={text} style={{ minHeight: '20vh'}} as="textarea" rows="3" name="text"/>
        </Form.Group>
      <Stack direction="horizontal" gap={3}>
        <Button disabled={loading} type="submit" size="lg">
          {loading? "Analyzing..." : "Submit"}
          </Button>
        <Button variant="danger" onClick={()=>reset()} size="lg">
          Clear
        </Button>
      </Stack>
    </Form>
    )
  }

  function reset() {
    updateText("")
    sentiments.forEach((s, i, arr) => sentiments[i].val[1](0))
  }

  function SentimentGroup(sentiment , i) {
    const elementRef = React.useRef();
    sentiments[i]['barRef'] = elementRef
    return (
      <Stack direction="horizontal" gap={3}>
        <p style={{align: "left", width: "100px"}}>{sentiment.labels[0].toUpperCase()}</p>
        <ProgressBar ref={elementRef} now={sentiment.val[0]} variant={sentiment.color} style={{width: '350px'}}/>
        <p style={{align: "right", width: "auto"}}>{sentiment.labels[1].toUpperCase()}</p>
      </Stack>
    )
  }

  function handleSubmit(input, queryFn = dummyQuery) {
    if (loading === false) {
      setLoading(true)
      //console.log(input)
      updateText(input)
      //queryFn(input, s.axis, s.val[1])
      /*querySentiment(input, queryFn).then(() => {
        setLoading(false);
      });*/
      sentiments.forEach((s, i, arr) => queryFn(input, s.axis, s.val[1]))
      //sentiments.forEach((s, i, arr) => console.log(s.val[0]))
    }
  }

  React.useEffect(() => {
    //textRef.current = text;
    //console.log(text)
  });

  return (
    <div className="App">
      <div className='App-header'>
        <h1>Decoji</h1>
        <Card style={{ width: '600px' }} border="primary">
          <Card.Body>
            <TextInput />
          </Card.Body>
          <Card.Footer>
              <Stack id="bars "gap={3}>
                {sentiments.map(SentimentGroup)}
              </Stack>
          </Card.Footer>
        </Card>
      </div>
    </div>
  );
}

export default App;
