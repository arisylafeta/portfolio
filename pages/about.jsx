import style from "../styles/About.module.css"

const AboutPage = () => {
  return (
    <>
      <h3>A Little Bit About Me</h3>
      <p className= {style.about}>Enthusiastic CS student with +2 years of working experience in web development and product management.
         Cofounder of Greenify, a web application startup that aggregates ESG data for sustainable investing. 
         Eager to contribute in all aspects of the software development lifecycle, from customer discovery through to development and delivery. 
         Seeking to expand body of knowledge by working in innovative and agile environments.
      </p>
    </>
  );
};

export async function getStaticProps() {
  return {
    props: { title: 'About' },
  };
}

export default AboutPage;
